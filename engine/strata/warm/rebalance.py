"""Deterministic freeze-only critical-path placement. No devices or routing changes.

Marginal profiles use sum_layer max(E[lane completion]), explicitly a surrogate.
Optional bounded co-selection samples evaluate mean(sum_layer max(completion)).
All times are caller-supplied measured service costs; no IQ2_XS time is guessed.
"""
import hashlib
import math
from pathlib import Path

LANES=('cpu','hip','sycl')
SUPPORTED={'IQ2_XXS','IQ2_XS','IQ1_M'}
GRANULE=2**21
MIN_PAYLOAD=20_000_000_000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded(n):
    return (n+GRANULE-1)//GRANULE*GRANULE


def optimise(plan, profile, costs, *, max_steps=64, min_payload=MIN_PAYLOAD):
    raw=Path(profile).read_bytes()
    nl=plan['layers'];inventory=plan['warm_inventory'];ne=len(inventory)//nl
    if len(raw)!=4*nl*ne or costs.get('profile_sha256')!=sha(profile):
        raise ValueError('need exact target-only real decode profile and matching costs provenance')
    if costs.get('schema')!='strata-warm-costs-v1' or costs.get('scope')!='measured-decode-excluding-warmup':
        raise ValueError('need measured decode costs excluding warmup')
    if not costs.get('evidence') or not isinstance(costs.get('decode_tokens'),int) or costs['decode_tokens']<=0:
        raise ValueError('need measured cost evidence and positive decode token count')
    if len(costs['layers'])!=nl:raise ValueError('cost layer extent')
    counts=[int.from_bytes(raw[i:i+4],'little') for i in range(0,len(raw),4)]
    experts={};assignment={}
    for i,row in enumerate(inventory):
        l,e,size,payload,*types=row
        if (l,e)!=divmod(i,ne) or len(types)!=3 or size<=0 or payload<=0 or payload>size:
            raise ValueError('invalid native expert inventory')
        experts[l,e]={'size':size,'payload':payload,'supported':set(types)<=SUPPORTED,'signature':'/'.join(types)}
        assignment[l,e]=0
    for row in plan['hip_resident']['experts']:assignment[tuple(row[:2])]=1
    for row in plan['warm_sycl']['members']:
        key=tuple(row)
        if assignment[key]:raise ValueError('overlapping native placement')
        assignment[key]=2
    hip_room=plan['warm_hip_room_bytes'];sycl_room=plan['warm_sycl']['reservation_bytes']
    # The native planner's complete footprint includes fixed + workspace + bank
    # reservation. It must satisfy both physical caps, before any optimisation.
    if plan['sycl_footprint']>28_500_000_000 or plan['hip_footprint']>30_450_647_040:
        raise ValueError('native fixed/workspace footprint exceeds device cap')
    service={};issue={}
    for l in range(nl):
        for lane,name in enumerate(LANES):
            entry=costs['layers'][l][name];issue[l,lane]=float(entry['issue_ns'])
            if not math.isfinite(issue[l,lane]) or issue[l,lane]<0:raise ValueError('invalid issue cost')
            for e in range(ne):
                key=(l,e);spec=entry.get('signatures',{}).get(experts[key]['signature'],entry.get('per_expert_ns'))
                if spec is None and (lane!=2 or experts[key]['supported']):raise ValueError('missing measured matrix-signature cost')
                value=float(spec) if spec is not None else 0.0
                if not math.isfinite(value) or value<=0 and (lane!=2 or experts[key]['supported']):raise ValueError('invalid service cost')
                service[l,e,lane]=value
    samples=costs.get('samples')
    demands=[[] for _ in range(nl)]
    if samples is None:
        for l in range(nl):demands[l]=[(tuple((e,counts[l*ne+e]/costs['decode_tokens']) for e in range(ne) if counts[l*ne+e]),1.0)]
        objective_kind='sum-layer-max-mean-completion; marginal-profile-surrogate'
    else:
        if not 1<=len(samples)<=4096:raise ValueError('trace must contain 1..4096 bounded co-selection samples')
        histogram=[0]*(nl*ne);tokens=0
        for sample in samples:
            weight=sample['weight'];selected=sample['layers']
            if type(weight) is not int or weight<=0 or len(selected)!=nl:raise ValueError('invalid trace sample')
            tokens+=weight
            for l,row in enumerate(selected):
                if len(row)!=len(set(row)) or any(type(e) is not int or not 0<=e<ne for e in row):raise ValueError('invalid trace selections')
                for e in row:histogram[l*ne+e]+=weight
                demands[l].append((tuple((e,1.0) for e in row),weight))
        if tokens!=costs['decode_tokens'] or histogram!=counts:raise ValueError('trace does not reproduce exact decode profile')
        demands=[[(row,w/tokens) for row,w in layer] for layer in demands]
        objective_kind='mean-sum-layer-max-completion; measured-co-selection-profile'

    states=[];occurrences={key:[] for key in experts}
    for l,layer in enumerate(demands):
        rows=[]
        for i,(selected,weight) in enumerate(layer):
            finish=[0.0]*3;active=[0]*3
            for e,n in selected:
                lane=assignment[l,e];finish[lane]+=n*service[l,e,lane];active[lane]+=1
                occurrences[l,e].append((i,n))
            rows.append([finish,active,weight])
        states.append(rows)
    def completion(l,finish,active):
        return max(finish[k]+(issue[l,k] if active[k] else 0) for k in range(3))
    current=[sum(w*completion(l,f,a) for f,a,w in rows) for l,rows in enumerate(states)]
    before_layers=current.copy();before=sum(current)
    def affected(changes):
        touched={}
        for key,lane in changes.items():
            old=assignment[key];l,e=key
            for i,n in occurrences[key]:
                loc=(l,i)
                if loc not in touched:
                    f,a,w=states[l][i];touched[loc]=[f.copy(),a.copy(),w]
                f,a,_=touched[loc];f[old]-=n*service[l,e,old];f[lane]+=n*service[l,e,lane];a[old]-=1;a[lane]+=1
        return touched
    def delta(changes):
        return sum(w*(completion(l,f,a)-completion(l,*states[l][i][:2])) for (l,i),(f,a,w) in affected(changes).items())

    sizes=[[0]*nl for _ in LANES];payload=[0]*3;coverage=[0]*nl
    for key,lane in assignment.items():
        l,e=key;sizes[lane][l]+=experts[key]['size'];payload[lane]+=experts[key]['payload'];coverage[l]+=lane==2
    def legal(changes,require_floor=True):
        hs=sizes[1].copy();ss=sum(sizes[2]);pp=payload.copy();cv=coverage.copy()
        for key,lane in changes.items():
            old=assignment[key];l,e=key;x=experts[key]
            if lane==2 and not x['supported']:return False
            if old==1:hs[l]-=x['size']
            if lane==1:hs[l]+=x['size']
            ss+=x['size']*((lane==2)-(old==2));pp[old]-=x['payload'];pp[lane]+=x['payload'];cv[l]+=(lane==2)-(old==2)
        if sum(map(rounded,hs))>hip_room or ss>sycl_room or min(pp[1:])<min_payload:return False
        if not require_floor and any(coverage[l]>0 and cv[l]==0 for l in range(nl)):return False
        if require_floor and any(cv[l]==0 and any(experts[l,e]['supported'] for e in range(ne)) for l in range(nl)):return False
        return True

    if not legal({},False):raise ValueError("native seed bank violates capacity/payload accounting")
    moves=[]
    def apply(changes,reason):
        touched=affected(changes);change_delta=delta(changes)
        for (l,i),row in touched.items():
            current[l]+=row[2]*(completion(l,*row[:2])-completion(l,*states[l][i][:2]));states[l][i]=row
        for key,lane in changes.items():
            old=assignment[key];l,e=key;x=experts[key];sizes[old][l]-=x['size'];sizes[lane][l]+=x['size'];payload[old]-=x['payload'];payload[lane]+=x['payload'];coverage[l]+=(lane==2)-(old==2);assignment[key]=lane
        moves.append({'reason':reason,'delta_ns':change_delta,'changes':[[*key,lane] for key,lane in sorted(changes.items())]})
    # Floors require an actual beneficial exchange, never an unlabelled objective
    # increase. Otherwise report the conflict and leave this freeze uncreated.
    for l in range(nl):
        if coverage[l] or not any(experts[l,e]['supported'] for e in range(ne)):continue
        best=None
        incoming=sorted(((l,e) for e in range(ne) if experts[l,e]['supported']),key=lambda key:(-counts[key[0]*ne+key[1]],key))[:16]
        outgoing=sorted((key for key,lane in assignment.items() if lane==2 and coverage[key[0]]>1),key=lambda key:(counts[key[0]*ne+key[1]],key))[:64]
        for a in incoming:
            for b in outgoing:
                proposals=[{a:2,b:assignment[a]}]
                if assignment[a]==0:
                    extra=[x for x in outgoing if x!=b and coverage[x[0]]>2][:2]
                    for n in (1,2):
                        if len(extra)>=n:proposals.append({a:2,b:0,**{x:0 for x in extra[:n]}})
                for c in proposals:
                    if not legal(c,False):continue
                    value=delta(c)
                    if value<=0 and (best is None or (value,a,b)<best[:3]):best=(value,a,b,c)
        if best is None:raise ValueError(f'B65 layer {l} coverage conflicts with capacities/objective; no freeze')
        apply(best[3],'coverage-beneficial-exchange')
    # Bounded deterministic coordinate descent: high/low-demand candidates in
    # every layer/lane. Evaluate all destination moves and a bounded eviction
    # shortlist; local optimum only, no global-optimum claim.
    for _ in range(max_steps):
        candidates=[]
        for l in range(nl):
            for lane in range(3):
                keys=sorted(((l,e) for e in range(ne) if assignment[l,e]==lane),key=lambda key:(counts[l*ne+key[1]],key))
                candidates+=keys[:2]+keys[-2:]
        candidates=sorted(set(candidates));best=None
        for a in candidates:
            for lane in range(3):
                if lane==assignment[a] or lane==2 and not experts[a]['supported']:continue
                # Cross-layer evictions let hot XS enter either GPU while
                # returning lower-value experts to the source tier.
                evictions=sorted((b for b in candidates if assignment[b]==lane),key=lambda b:(counts[b[0]*ne+b[1]]*service[b[0],b[1],lane],b))[:12]
                proposals=[{a:lane}]+[{a:lane,b:assignment[a]} for b in evictions]
                # Different raw signatures can need more than one eviction.
                for count in (2,3):
                    if len(evictions)>=count:proposals.append({a:lane,**{b:0 for b in evictions[:count]}})
                old=assignment[a]
                if lane and old==0:
                    other=3-lane
                    third=sorted((b for b in candidates if assignment[b]==other),key=lambda b:(counts[b[0]*ne+b[1]],b))[:4]
                    proposals += [{a:lane,b:other,c:0} for b in evictions[:4] for c in third]
                elif lane and old:
                    third=sorted((b for b in candidates if assignment[b]==0),key=lambda b:(-counts[b[0]*ne+b[1]],b))[:4]
                    proposals += [{a:lane,b:0,c:old} for b in evictions[:4] for c in third]
                for c in proposals:
                    if not legal(c):continue
                    value=delta(c)
                    if value < -1e-6 and (best is None or (value,tuple(sorted(c.items())))<best[:2]):best=(value,tuple(sorted(c.items())),c)
        if best is None:break
        apply(best[2],'critical-path-coordinate-descent')
    # Maximise admitted capacity, including cold entries. Refill can expose a
    # critical lane, so require non-increasing objective for every addition.
    for lane in (1,2):
        for key in sorted(assignment,key=lambda key:(-counts[key[0]*ne+key[1]]/experts[key]['size'],key)):
            if assignment[key]!=0:continue
            c={key:lane}
            if legal(c) and delta(c)<=1e-6:apply(c,'nonincreasing-capacity-refill')
    # Native refuses refillable slack as well. Detect conflicts before writing.
    for key,lane in assignment.items():
        if lane:continue
        for dest in (1,2):
            if legal({key:dest}):raise ValueError('maximal bank capacity conflicts with objective; no freeze')
    if not legal({}) or sum(current)>before+1e-6:raise ValueError('rebalance violated objective/capacity')
    return assignment,{'schema':'strata-warm-rebalance-v1','objective':objective_kind,'before_ns':before,'after_ns':sum(current),
        'per_layer_before_ns':before_layers,
        'per_layer_after_ns':current,'profile_sha256':sha(profile),'costs':costs,'moves':moves,
        'hip_driver_charge_bytes':sum(map(rounded,sizes[1])),'sycl_allocated_bytes':sum(sizes[2]),
        'hip_room_bytes':hip_room,'sycl_room_bytes':sycl_room,'payload_bytes':dict(zip(LANES,payload)),
        'b65_experts_per_layer':coverage,'limitation':'bounded local search; service model estimates, native performance unmeasured'}


def assignment_text(assignment,plan):
    nl=plan['layers'];ne=len(assignment)//nl
    return f"STRATA_WARM_PLACEMENT_V1 {nl} {ne} {plan['warm_sycl']['profile_tag']}\n"+''.join(f'{l} {e} {lane}\n' for (l,e),lane in sorted(assignment.items()))
