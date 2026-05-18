#!/usr/bin/env python
"""Simple import test for the restructured app."""

import os
os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/Dataset_management"

import sys
sys.path.insert(0, r"d:\FIXED_WING_DATASET_FEB_5_2026\Roboflow_dataset_formate_codes\dataset_backend")

try:
    from app.main import app
    print("SUCCESS: Main app imported successfully")
    print(f"App title: {app.title}")
    
    # List registered routes
    route_paths = set()
    for route in app.routes:
        if hasattr(route, 'path'):
            route_paths.add(route.path)
    
    print(f"Registered {len(route_paths)} unique paths:")
    for p in sorted(route_paths):
        print(f"  {p}")
        
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
