import json, sys
for line in open(sys.argv[1]):
    if '"step"' in line:
        r = json.loads(line[line.index("{"):])
        c, b, o = r["closed"], r.get("bad policy, closed loop"), r.get("bad policy, open loop")
        print(f"step {r['step']:4d} loss {r.get('loss')}")
        print(f"   open loop  expert E1/E8/E24/E100 {list(r['expert'].values())}  mixed {list(r['mixed'].values())}")
        print(f"   walk test  fell {c['fell']:.3f} track {c['track/step']:.4f} u_bar {c['u_bar']:.3f} tilt>30 {c['robots ever tilt>30']:.3f}")
        if b: print(f"   fall test  fell {b['fell']:.3f} track {b['track/step']:.4f} len {b['len']:.0f} u_bar {b['u_bar']:.3f} tilt>30 {b['robots ever tilt>30']:.3f} | open loop E1/E8/E24 {o['E1']}/{o['E8']}/{o['E24']} tilt true/pred {o['tilt at end: true / predicted (median deg)']}")
    elif "sec/step" in line: print(line.strip()[:120])
