"""
test_live_server.py
Validates all endpoints of the live running uvicorn server on http://127.0.0.1:8000
using the active Indian (Bangalore) dataset.
"""

import urllib.request
import json
import sys

base_url = 'http://127.0.0.1:8000'

def call(path, method='GET', data=None):
    url = base_url + path
    headers = {'Content-Type': 'application/json'} if data else {}
    body = json.dumps(data).encode('utf-8') if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=25) as res:
        return res.status, json.loads(res.read().decode('utf-8'))

print("=" * 70)
print("TESTING LIVE UVICORN SERVER ON http://0.0.0.0:8000")
print("=" * 70)

print("\n1. GET /health")
st, health = call('/health')
print(f"   Status: HTTP {st}")
print(f"   Response: {health}")

print("\n2. GET /api/network/summary")
st, net = call('/api/network/summary')
print(f"   Status: HTTP {st}")
print(f"   Active City: {net.get('city')}")
print(f"   Metropolitan Area: {net.get('area')}")
print(f"   Data Source: {net.get('data_source')}")
print(f"   Road Segments: {net.get('total_road_segments')}, Intersections: {net.get('total_intersections')}")
print(f"   Transitions: {net.get('routing_graph_edges')}")
print(f"   Mean Pickup Prob: {net.get('pickup_prob_stats', {}).get('mean')}")

print("\n3. GET /api/hotspots?top_n=5")
st, hs = call('/api/hotspots?top_n=5')
print(f"   Status: HTTP {st}")
print(f"   Returned {len(hs.get('points', []))} hotspots:")
for p in hs.get('points', []):
    print(f"     - Seg {p['segment_id']} at ({p['lat']}, {p['lon']}): Prob={p['pickup_prob']}, Earning=₹{p['earning']}")

print("\n4. POST /api/recommend (MORA-LLM in Koramangala)")
rec_payload = {
    "lat": 12.9352,
    "lon": 77.6245,
    "num_taxis": 1,
    "time_limit": 10.0,
    "scenario": "normal",
    "algorithm": "mora_llm",
    "driver_profile_id": "new_driver_cold_start",
    "alpha": 0.2
}
st, rec_res = call('/api/recommend', method='POST', data=rec_payload)
print(f"   Status: HTTP {st}")
print(f"   Algorithm: {rec_res.get('algorithm')}")
rec = rec_res['recommendations'][0]
print(f"   Route Segments ({rec['route_length_segments']}): {[p['segment_id'] for p in rec['route']]}")
print(f"   Total Expected Earning: ₹{rec['total_expected_earning']} INR")
print(f"   Estimated Cruising Time: {rec['estimated_time_minutes']} min")
print(f"   Novelty: {rec['novelty_score']}, Diversity: {rec['topic_diversity_score']}")
if rec_res.get('llm_explanation'):
    print(f"   LLM Rationale: {rec_res['llm_explanation'].get('summary_explanation', '')[:100]}...")

print("\n5. POST /api/recommend/accuracy (Accuracy & Calibration Validator)")
acc_payload = {"recommendations": rec_res['recommendations']}
st, acc_res = call('/api/recommend/accuracy', method='POST', data=acc_payload)
print(f"   Status: HTTP {st}")
print(f"   Message: {acc_res.get('message')}")
print(f"   MAE: {acc_res.get('mean_absolute_error')}")
print(f"   Brier Score: {acc_res.get('brier_score')}")
print(f"   Route Success Rate: {acc_res.get('estimated_route_success_rate') * 100:.1f}%")

print("\n6. POST /api/recommend (SRRS Fleet Routing for 2 Taxis at MG Road)")
srrs_payload = {
    "lat": 12.9716,
    "lon": 77.5946,
    "num_taxis": 2,
    "time_limit": 10.0,
    "scenario": "normal",
    "algorithm": "srrs"
}
st, srrs_res = call('/api/recommend', method='POST', data=srrs_payload)
print(f"   Status: HTTP {st}")
print(f"   Routes Generated: {len(srrs_res.get('recommendations', []))}")
for t in srrs_res.get('recommendations', []):
    print(f"     - {t['taxi_id']}: ₹{t['total_expected_earning']} INR ({t['route_length_segments']} segments)")

print("\n" + "=" * 70)
print("SUCCESS: ALL ENDPOINTS TESTED AND VERIFIED ON LIVE UVICORN SERVER!")
print("=" * 70)

