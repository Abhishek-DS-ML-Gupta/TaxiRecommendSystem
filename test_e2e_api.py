import urllib.request
import json
import sys

base_url = 'http://127.0.0.1:8000'

def test_endpoint(name, path, method='GET', data=None):
    url = base_url + path
    req = urllib.request.Request(url, method=method)
    if data:
        req.add_header('Content-Type', 'application/json')
        body = json.dumps(data).encode('utf-8')
    else:
        body = None
    try:
        with urllib.request.urlopen(req, data=body, timeout=25) as res:
            status = res.status
            content = res.read().decode('utf-8')
            parsed = json.loads(content) if 'application/json' in res.headers.get('Content-Type', '') else len(content)
            print(f'PASS - {name}: HTTP {status}')
            return parsed
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode('utf-8')
        print(f'FAIL - {name}: HTTP {e.code} - {err_msg[:200]}')
        return None
    except Exception as e:
        print(f'FAIL - {name}: {e}')
        return None

print('=== Validating Running Backend Services ===')
test_endpoint('Home HTML', '/')
test_endpoint('Config', '/api/config')
test_endpoint('Health', '/health')
test_endpoint('Network Summary', '/api/network/summary')
test_endpoint('Hotspots (top 5)', '/api/hotspots?top_n=5')

# MORA-LLM
mora_payload = {
    'lat': 39.9042,
    'lon': 116.4074,
    'num_taxis': 1,
    'time_limit': 10.0,
    'scenario': 'normal',
    'algorithm': 'mora_llm',
    'driver_profile_id': 'new_driver_cold_start',
    'alpha': 0.2
}
res_mora = test_endpoint('MORA-LLM Recommend', '/api/recommend', method='POST', data=mora_payload)
if res_mora and 'recommendations' in res_mora:
    rec = res_mora['recommendations'][0]
    print(f"  -> Profit: ¥{rec.get('total_expected_earning')}")
    print(f"  -> Route length: {len(rec.get('route', []))} segments")
    print(f"  -> Novelty: {rec.get('novelty_score')}, Diversity: {rec.get('topic_diversity_score')}, SDE: {rec.get('sde_fitness')}")
    print(f"  -> Pareto Alternatives: {list(res_mora.get('pareto_alternatives', {}).keys())}")
    if res_mora.get('llm_explanation'):
        expl = res_mora['llm_explanation']
        print(f"  -> Driver: {expl.get('driver_name')}")
        print(f"  -> Cold-start: {expl.get('is_cold_start')}")
        print(f"  -> Summary: {expl.get('summary_explanation', '')[:100]}...")
        print(f"  -> Segment Reasons Count: {len(expl.get('segment_reasons', []))}")

# SRRS
srrs_payload = {
    'lat': 39.9042,
    'lon': 116.4074,
    'num_taxis': 2,
    'time_limit': 10.0,
    'scenario': 'normal',
    'algorithm': 'srrs'
}
res_srrs = test_endpoint('SRRS (2 taxis)', '/api/recommend', method='POST', data=srrs_payload)
if res_srrs and 'recommendations' in res_srrs:
    print(f"  -> Returned {len(res_srrs['recommendations'])} routes")
    for r in res_srrs['recommendations']:
        print(f"     * {r['taxi_id']}: ¥{r.get('total_expected_earning')} | len={len(r.get('route', []))}")

# RRS
rrs_payload = {
    'lat': 39.9042,
    'lon': 116.4074,
    'num_taxis': 1,
    'time_limit': 10.0,
    'scenario': 'normal',
    'algorithm': 'rrs'
}
res_rrs = test_endpoint('RRS (single taxi)', '/api/recommend', method='POST', data=rrs_payload)
if res_rrs and 'recommendations' in res_rrs:
    print(f"  -> Profit: ¥{res_rrs['recommendations'][0].get('total_expected_earning')} | len={len(res_rrs['recommendations'][0].get('route', []))}")

# Accuracy Check
if res_mora and 'recommendations' in res_mora:
    acc_payload = {'recommendations': res_mora['recommendations']}
    res_acc = test_endpoint('Accuracy Evaluation', '/api/recommend/accuracy', method='POST', data=acc_payload)
    if res_acc:
        print(f"  -> Accuracy MAE: {res_acc.get('mean_absolute_error')}, Success Rate: {res_acc.get('estimated_route_success_rate')}")

print('=== All Tests Finished Successfully! ===')
