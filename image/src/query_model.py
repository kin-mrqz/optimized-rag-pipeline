import os
import time
import boto3
import uuid
from pydantic import BaseModel, Field
from typing import Optional, List
from botocore.exceptions import ClientError

TABLE_NAME = os.environ.get("TABLE_NAME", "RagQueryTable")


class QueryModel(BaseModel):
    query_id: str = Field(
        default_factory = lambda: uuid.uuid4().hex,
    )
    create_time: int = Field(
        default_factory = lambda: int(time.time())
    )
    query_text: str
    answer_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list) 
    # note: retreived document sources not yet enabled in RAG feature
    is_complete: bool = False


    # Class method to get the DynamoDB table resource
    @classmethod
    def get_table(cls: "QueryModel") -> boto3.resource:
        """Get the DynamoDB table resource."""
        dynamodb = boto3.resource("dynamodb")
        return dynamodb.Table(TABLE_NAME)
    
    def put_item(self):
        item = self.as_ddb_item()
        try:
            response = QueryModel.get_table().put_item(Item=item)
            print(response)
        except ClientError as e:
            print(f"Error putting item in DynamoDB: {e.response['Error']['Message']}")
            raise e
        
    def as_ddb_item(self):
        item = {k: v for k, v in self.dict().items() if v is not None}
        return item
    
    
    # Helper function to convert a DynamoDB item to a QueryModel instance
    @classmethod
    def get_item(cls: "QueryModel", query_id: str) -> "QueryModel":
        try:
            response = cls.get_table().get_item(Key = {"query_id": query_id})
        except ClientError as e:
            print(f"Error getting item from DynamoDB: {e.response['Error']['Message']}")
            return None
        
        if "Item" in response:
            item = response["Item"]
            return cls(**item)
        else:
            return None