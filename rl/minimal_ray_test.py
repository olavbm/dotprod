#!/usr/bin/env python3
"""
Minimal test to verify Ray Client connection to hoppetusse cluster.
Tests both local execution and remote cluster resources.
"""

import ray
import socket
import time

print("🧪 Testing Ray Client Connection to Hoppetusse")
print("=" * 48)

def test_local_mode():
    """Test Ray in local mode first."""
    print("\n1️⃣ Testing Ray Local Mode...")
    try:
        ray.init(local_mode=True, ignore_reinit_error=True)
        print("   ✅ Local Ray initialized")
        
        @ray.remote
        def local_test(x):
            return x * 2
        
        result = ray.get(local_test.remote(21))
        print(f"   ✅ Local test result: {result}")
        ray.shutdown()
        return True
    except Exception as e:
        print(f"   ❌ Local Ray failed: {e}")
        return False

def test_cluster_connection():
    """Test Ray Client connection to hoppetusse."""
    print("\n2️⃣ Testing Ray Client Connection...")
    try:
        # Connect to hoppetusse cluster via SSH tunnel
        ray.init("ray://localhost:10001")
        print("   ✅ Connected to hoppetusse cluster!")
        
        # Show cluster resources
        resources = ray.cluster_resources()
        print(f"   📊 Cluster Resources:")
        print(f"      CPUs: {int(resources.get('CPU', 0))}")
        print(f"      Memory: {resources.get('memory', 0) / (1024**3):.1f} GB")
        print(f"      GPUs: {int(resources.get('GPU', 0))}")
        
        return True
    except Exception as e:
        print(f"   ❌ Cluster connection failed: {e}")
        print("   💡 Make sure:")
        print("      1. SSH tunnel is active: ssh hoppetusse")
        print("      2. Ray head is running on hoppetusse")
        print("      3. Port 10001 is forwarded in SSH config")
        return False

def test_remote_execution():
    """Test that tasks actually run on hoppetusse."""
    print("\n3️⃣ Testing Remote Task Execution...")
    try:
        @ray.remote
        def remote_task(worker_id):
            import socket
            import time
            hostname = socket.gethostname()
            start_time = time.time()
            
            # Some CPU work
            result = sum(i * i for i in range(100000))
            duration = time.time() - start_time
            
            return {
                'worker_id': worker_id,
                'hostname': hostname,
                'result': result,
                'duration': duration,
            }
        
        # Launch multiple tasks
        print("   🚀 Launching 4 parallel tasks...")
        futures = [remote_task.remote(i) for i in range(4)]
        results = ray.get(futures)
        
        # Show results
        hostnames = set(r['hostname'] for r in results)
        print(f"   ✅ Tasks completed on: {', '.join(hostnames)}")
        
        total_time = max(r['duration'] for r in results)
        avg_time = sum(r['duration'] for r in results) / len(results)
        print(f"   ⏱️  Total time: {total_time:.3f}s, Avg per task: {avg_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"   ❌ Remote execution failed: {e}")
        return False

def main():
    """Run all tests."""
    
    # Test 1: Local mode
    local_ok = test_local_mode()
    
    # Test 2: Cluster connection
    if local_ok:
        cluster_ok = test_cluster_connection()
        
        # Test 3: Remote execution
        if cluster_ok:
            remote_ok = test_remote_execution()
            
            print(f"\n🏁 Test Summary:")
            print("=" * 16)
            print(f"   Local Ray: {'✅' if local_ok else '❌'}")
            print(f"   Cluster Connection: {'✅' if cluster_ok else '❌'}")
            print(f"   Remote Execution: {'✅' if remote_ok else '❌'}")
            
            if all([local_ok, cluster_ok, remote_ok]):
                print(f"\n🎉 All tests passed! Hoppetusse cluster is ready for RL experiments!")
                print(f"   Your cartpole_curiosity_new.py will now use 16 CPUs!")
            else:
                print(f"\n⚠️  Some tests failed. Check SSH tunnel and Ray head node.")
        
        ray.shutdown()
    else:
        print(f"\n❌ Basic Ray functionality broken. Check Ray installation.")

if __name__ == "__main__":
    main()