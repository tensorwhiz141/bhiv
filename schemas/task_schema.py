from pydantic import BaseModel
from typing import List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class TaskSchema(BaseModel):
    task: str
    data: str
    model: str
    keywords: Optional[List[str]] = []

    def validate_task(self):
        logger.info(f"Validating task: {self.task}")
        if not self.task or not self.data:
            logger.error("Task or data cannot be empty")
            raise ValueError("Task or data cannot be empty")
        return self

if __name__ == "__main__":
    task = TaskSchema(task="summarize", data="Sample PDF", model="edumentor_agent", keywords=["summarize"])
    print(task.validate_task())




# from pydantic import BaseModel
# from typing import List, Optional
# from utils.logger import get_logger

# logger = get_logger(__name__)

# class TaskSchema(BaseModel):
#     task: str
#     data: str
#     model: str
#     keywords: Optional[List[str]] = []

#     def validate_task(self):
#         logger.info(f"Validating task: {self.task}")
#         if not self.task or not self.data:
#             logger.error("Task or data cannot be empty")
#             raise ValueError("Task or data cannot be empty")
#         return self

# if __name__ == "__main__":
#     task = TaskSchema(task="summarize", data="Sample PDF", model="llama", keywords=["summarize"])
#     print(task.validate_task())