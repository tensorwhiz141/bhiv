import logging
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

class Calculator:
    """Simple calculator tool for basic arithmetic."""
    def evaluate(self, expression: str) -> Dict[str, Any]:
        """Evaluate a mathematical expression."""
        try:
            from operator import add, sub, mul, truediv
            ops = {'+': add, '-': sub, '*': mul, '/': truediv}
            parts = expression.replace(' ', '').split()
            if len(parts) != 3:
                raise ValueError("Expression must be in format: number operator number")
            num1, op, num2 = float(parts[0]), parts[1], float(parts[2])
            if op not in ops:
                raise ValueError(f"Unsupported operator: {op}")
            result = ops[op](num1, num2)
            logger.info(f"Calculated {expression} = {result}")
            return {
                "result": str(result),
                "status": 200,
                "keywords": ["calculation"]
            }
        except Exception as e:
            logger.error(f"Error evaluating expression {expression}: {str(e)}")
            return {"error": f"Calculation failed: {str(e)}", "status": 400, "keywords": []}

if __name__ == "__main__":
    calc = Calculator()
    print(calc.evaluate("5 + 3"))