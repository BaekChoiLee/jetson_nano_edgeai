import json, sys

with open(sys.argv[1]) as f:
    old = json.load(f)
with open(sys.argv[2]) as f:
    new = json.load(f)

print("=" * 80)
print("  OLD vs NEW comparison (latency & accuracy)")
print("=" * 80)

for model in new['models']:
    print(f"\n--- {model} ---")
    for rt in new['runtimes']:
        o = old['matrix'].get(model, {}).get(rt, {})
        n = new['matrix'].get(model, {}).get(rt, {})
        if not n:
            continue
        
        diffs = []
        
        # Latency MAXN
        om = o.get('latency_MAXN_mean_ms')
        nm = n.get('latency_MAXN_mean_ms')
        if om and nm:
            pct = (nm - om) / om * 100
            if abs(pct) > 5:
                diffs.append(f"10W: {om:.1f} -> {nm:.1f} ({pct:+.1f}%)")
        
        # Latency 5W
        o5 = o.get('latency_5W_mean_ms')
        n5 = n.get('latency_5W_mean_ms')
        if o5 and n5:
            pct5 = (n5 - o5) / o5 * 100
            if abs(pct5) > 5:
                diffs.append(f"5W: {o5:.1f} -> {n5:.1f} ({pct5:+.1f}%)")
        
        # Accuracy top1
        ot1 = o.get('accuracy_top1')
        nt1 = n.get('accuracy_top1')
        if ot1 != nt1:
            ot1s = str(ot1) if ot1 is not None else 'MISS'
            nt1s = str(nt1) if nt1 is not None else 'MISS'
            diffs.append(f"top1: {ot1s} -> {nt1s}")
        
        # Accuracy map50
        om50 = o.get('accuracy_map_50')
        nm50 = n.get('accuracy_map_50')
        if om50 != nm50:
            om50s = f"{om50:.4f}" if om50 is not None else 'MISS'
            nm50s = f"{nm50:.4f}" if nm50 is not None else 'MISS'
            diffs.append(f"mAP50: {om50s} -> {nm50s}")
        
        # Status change
        os_stat = o.get('accuracy_status')
        ns_stat = n.get('accuracy_status')
        if os_stat != ns_stat:
            diffs.append(f"acc_status: {os_stat} -> {ns_stat}")
        
        if diffs:
            print(f"  {rt}: {' | '.join(diffs)}")

print(f"\n{'=' * 80}")
print(f"  OLD missing accuracy: {old['summary']}")
print(f"  NEW missing accuracy: {new['summary']}")
print(f"{'=' * 80}")
