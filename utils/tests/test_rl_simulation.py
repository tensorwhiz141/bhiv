import requests
import uuid
import json
import os
import time
from typing import Dict, List, Any
from utils.logger import get_logger
from reinforcement.replay_buffer import replay_buffer
from reinforcement.reward_functions import get_reward_from_output
from config.settings import RL_CONFIG

logger = get_logger(__name__)

def create_test_files():
    """Create test files for simulation if they don't exist."""
    test_files = {
        "test.pdf": "Sample PDF content for testing",
        "test2.pdf": "Another PDF content for testing",
        "test_image.jpg": "test_image.jpg",  # Existing file
        "test.wav": "test.wav"  # Existing file
    }

    # Note: For actual testing, these files should exist in the workspace
    return test_files

def run_mixed_input_simulation():
    """Simulate mixed input types (PDF + image + audio) in single runs."""
    mixed_tasks = [
        {
            "task": "multi_analyze",
            "files": [
                {"path": "test.pdf", "type": "pdf", "data": "Educational PDF content about AI"},
                {"path": "test_image.jpg", "type": "image", "data": "AI diagram image"},
                {"path": "test.wav", "type": "audio", "data": "Audio lecture about AI"}
            ],
            "model": "edumentor_agent"
        },
        {
            "task": "multi_summarize",
            "files": [
                {"path": "test2.pdf", "type": "pdf", "data": "Research paper on machine learning"},
                {"path": "test2.png", "type": "image", "data": "ML flowchart image"}
            ],
            "model": "edumentor_agent"
        }
    ]

    results = []

    for task in mixed_tasks:
        task_id = str(uuid.uuid4())
        logger.info(f"Running mixed input task {task_id} with {len(task['files'])} files")

        try:
            # Simulate multi-input processing by calling handle_multi_task endpoint
            response = requests.post(
                "http://localhost:8002/handle_multi_task",
                json={
                    "agent": task["model"],
                    "files": task["files"],
                    "task_type": task["task"]
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            reward = get_reward_from_output(result.get("agent_output", {}), task_id)

            results.append({
                "task_id": task_id,
                "task_type": task["task"],
                "file_count": len(task["files"]),
                "file_types": [f["type"] for f in task["files"]],
                "result": result,
                "reward": reward,
                "processing_time": result.get("processing_time", 0)
            })

            logger.info(f"Mixed input task {task_id} completed with reward: {reward}")

        except Exception as e:
            logger.error(f"Mixed input task {task_id} failed: {str(e)}")
            results.append({
                "task_id": task_id,
                "error": str(e),
                "reward": 0.0,
                "file_count": len(task["files"]),
                "file_types": [f["type"] for f in task["files"]]
            })

    return results

def run_edge_case_simulation():
    """Test edge cases like invalid files and network failures."""
    edge_cases = [
        {
            "name": "invalid_file",
            "task": {"agent": "edumentor_agent", "input": "test", "pdf_path": "nonexistent.pdf", "input_type": "pdf"}
        },
        {
            "name": "empty_input",
            "task": {"agent": "edumentor_agent", "input": "", "input_type": "text"}
        },
        {
            "name": "invalid_agent",
            "task": {"agent": "nonexistent_agent", "input": "test content", "input_type": "text"}
        },
        {
            "name": "malformed_request",
            "task": {"invalid_field": "test"}
        }
    ]

    results = []

    for case in edge_cases:
        task_id = str(uuid.uuid4())
        logger.info(f"Testing edge case: {case['name']}")

        try:
            response = requests.post(
                "http://localhost:8002/handle_task",
                json=case["task"],
                timeout=10
            )

            # Even if the request succeeds, we want to check the response
            if response.status_code == 200:
                result = response.json()
                reward = get_reward_from_output(result.get("agent_output", {}), task_id)
            else:
                result = {"error": f"HTTP {response.status_code}", "status": response.status_code}
                reward = 0.0

            results.append({
                "case": case["name"],
                "task_id": task_id,
                "result": result,
                "reward": reward,
                "status_code": response.status_code
            })

        except requests.exceptions.Timeout:
            logger.warning(f"Edge case {case['name']} timed out")
            results.append({
                "case": case["name"],
                "task_id": task_id,
                "error": "timeout",
                "reward": 0.0
            })
        except Exception as e:
            logger.error(f"Edge case {case['name']} failed: {str(e)}")
            results.append({
                "case": case["name"],
                "task_id": task_id,
                "error": str(e),
                "reward": 0.0
            })

    return results

def run_simulation():
    """Enhanced simulation with mixed inputs and edge cases."""
    # Original single-input tasks
    tasks = [
        {"task": "summarize", "data": "Sample PDF content", "model": "edumentor_agent", "input_type": "pdf", "pdf_path": "test.pdf"},
        {"task": "summarize", "data": "Sample image content", "model": "edumentor_agent", "input_type": "image"},
        {"task": "summarize", "data": "Sample audio content", "model": "edumentor_agent", "input_type": "audio"},
        {"task": "summarize", "data": "Sample text content", "model": "edumentor_agent", "input_type": "text"},
        {"task": "calculate", "data": "5 + 3", "model": "edumentor_agent", "input_type": "text"},
        {"task": "summarize", "data": "Vedic wisdom", "model": "vedas_agent", "input_type": "text"},
        {"task": "summarize", "data": "Health advice", "model": "wellness_agent", "input_type": "text"},
        {"task": "summarize", "data": "Educational content", "model": "edumentor_agent", "input_type": "text"},
        {"task": "summarize", "data": "Another PDF", "model": "edumentor_agent", "input_type": "pdf", "pdf_path": "test2.pdf"},
        {"task": "summarize", "data": "Another text", "model": "edumentor_agent", "input_type": "text"}
    ]
    
    results = {
        "single_input": {"with_rl": [], "without_rl": []},
        "mixed_input": [],
        "edge_cases": [],
        "summary": {}
    }

    # Run single-input tasks with RL
    logger.info("Running single-input tasks with RL enabled")
    RL_CONFIG["use_rl"] = True
    for task in tasks:
        task_id = str(uuid.uuid4())
        try:
            response = requests.post(
                "http://localhost:8002/handle_task",
                json={"agent": task["model"], "input": task["data"], "pdf_path": task.get("pdf_path", ""), "input_type": task["input_type"]},
                timeout=15
            )
            response.raise_for_status()
            result = response.json()
            reward = get_reward_from_output(result.get("agent_output", {}), task_id)
            results["single_input"]["with_rl"].append({"task_id": task_id, "result": result, "reward": reward})
            logger.info(f"RL Task {task_id}: reward={reward}")
        except Exception as e:
            logger.error(f"RL Task {task_id} failed: {str(e)}")
            results["single_input"]["with_rl"].append({"task_id": task_id, "error": str(e), "reward": 0.0})

    # Run single-input tasks without RL
    logger.info("Running single-input tasks without RL")
    RL_CONFIG["use_rl"] = False
    for task in tasks:
        task_id = str(uuid.uuid4())
        try:
            response = requests.post(
                "http://localhost:8002/handle_task",
                json={"agent": task["model"], "input": task["data"], "pdf_path": task.get("pdf_path", ""), "input_type": task["input_type"]},
                timeout=15
            )
            response.raise_for_status()
            result = response.json()
            reward = get_reward_from_output(result.get("agent_output", {}), task_id)
            results["single_input"]["without_rl"].append({"task_id": task_id, "result": result, "reward": reward})
            logger.info(f"Non-RL Task {task_id}: reward={reward}")
        except Exception as e:
            logger.error(f"Non-RL Task {task_id} failed: {str(e)}")
            results["single_input"]["without_rl"].append({"task_id": task_id, "error": str(e), "reward": 0.0})

    # Run mixed-input simulation
    logger.info("Running mixed-input simulation")
    results["mixed_input"] = run_mixed_input_simulation()

    # Run edge case simulation
    logger.info("Running edge case simulation")
    results["edge_cases"] = run_edge_case_simulation()

    # Calculate summary statistics
    rl_rewards = [r["reward"] for r in results["single_input"]["with_rl"] if "reward" in r]
    no_rl_rewards = [r["reward"] for r in results["single_input"]["without_rl"] if "reward" in r]
    mixed_rewards = [r["reward"] for r in results["mixed_input"] if "reward" in r]

    results["summary"] = {
        "rl_avg_reward": sum(rl_rewards) / len(rl_rewards) if rl_rewards else 0,
        "no_rl_avg_reward": sum(no_rl_rewards) / len(no_rl_rewards) if no_rl_rewards else 0,
        "mixed_avg_reward": sum(mixed_rewards) / len(mixed_rewards) if mixed_rewards else 0,
        "rl_success_rate": len([r for r in results["single_input"]["with_rl"] if "error" not in r]) / len(results["single_input"]["with_rl"]),
        "no_rl_success_rate": len([r for r in results["single_input"]["without_rl"] if "error" not in r]) / len(results["single_input"]["without_rl"]),
        "mixed_success_rate": len([r for r in results["mixed_input"] if "error" not in r]) / len(results["mixed_input"]) if results["mixed_input"] else 0,
        "edge_case_handled": len([r for r in results["edge_cases"] if r.get("status_code", 0) != 500]) / len(results["edge_cases"]) if results["edge_cases"] else 0
    }

    # Save comprehensive results
    with open("logs/comprehensive_simulation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Log best runs as mentor logs
    best_runs = sorted(results["single_input"]["with_rl"], key=lambda x: x.get("reward", 0), reverse=True)[:5]
    with open("logs/mentor_logs.json", "w") as f:
        json.dump(best_runs, f, indent=2, default=str)

    logger.info("Saved comprehensive simulation results and mentor logs")
    logger.info(f"Summary: RL avg reward: {results['summary']['rl_avg_reward']:.3f}, "
                f"No-RL avg reward: {results['summary']['no_rl_avg_reward']:.3f}, "
                f"Mixed avg reward: {results['summary']['mixed_avg_reward']:.3f}")

    return results

def run_comprehensive_test():
    """Run all simulation types and return comprehensive results."""
    logger.info("Starting comprehensive RL simulation test")

    # Ensure test files exist (in a real scenario)
    create_test_files()

    # Run full simulation
    results = run_simulation()

    # Print summary
    print("\n=== COMPREHENSIVE RL SIMULATION RESULTS ===")
    print(f"Single Input - RL Average Reward: {results['summary']['rl_avg_reward']:.3f}")
    print(f"Single Input - No-RL Average Reward: {results['summary']['no_rl_avg_reward']:.3f}")
    print(f"Mixed Input Average Reward: {results['summary']['mixed_avg_reward']:.3f}")
    print(f"RL Success Rate: {results['summary']['rl_success_rate']:.1%}")
    print(f"No-RL Success Rate: {results['summary']['no_rl_success_rate']:.1%}")
    print(f"Mixed Input Success Rate: {results['summary']['mixed_success_rate']:.1%}")
    print(f"Edge Cases Handled: {results['summary']['edge_case_handled']:.1%}")

    print(f"\nMixed Input Tasks: {len(results['mixed_input'])}")
    for task in results['mixed_input']:
        if 'error' not in task:
            print(f"  Task {task['task_id'][:8]}: {task['file_count']} files ({', '.join(task['file_types'])}), reward: {task['reward']:.3f}")
        else:
            print(f"  Task {task['task_id'][:8]}: FAILED - {task['error']}")

    print(f"\nEdge Cases: {len(results['edge_cases'])}")
    for case in results['edge_cases']:
        status = "HANDLED" if case.get('status_code', 0) != 500 else "FAILED"
        print(f"  {case['case']}: {status}")

    return results

if __name__ == "__main__":
    results = run_comprehensive_test()

    # Also run individual tests for debugging
    print("\n=== INDIVIDUAL TEST RESULTS ===")
    print("Mixed Input Results:", len(results["mixed_input"]))
    print("Edge Case Results:", len(results["edge_cases"]))