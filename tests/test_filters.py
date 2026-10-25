# test_filters.py
from filters import TopKFilter, BlacklistFilter, CBFFilter, AdaptiveCBFFilter
print("All filter imports OK")
print(f"AdaptiveCBF alpha at h=0.5: {AdaptiveCBFFilter()._get_adaptive_alpha(0.5):.3f}")
print(f"AdaptiveCBF alpha at h=0.0: {AdaptiveCBFFilter()._get_adaptive_alpha(0.0):.3f}")
print(f"AdaptiveCBF alpha at h=-0.3: {AdaptiveCBFFilter()._get_adaptive_alpha(-0.3):.3f}")