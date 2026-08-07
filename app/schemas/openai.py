import time
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the author of this message (system, user, assistant)")
    content: str = Field(..., description="The contents of the message")
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = Field(default="llama-3-8b-instruct")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1)
    stream: bool = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=1024)
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)
    user: Optional[str] = None

    def extract_user_prompt(self) -> str:
        """Extracts the last user prompt from the messages list."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""

    def extract_system_prompt(self) -> str:
        """Extracts the system prompt if present."""
        for msg in self.messages:
            if msg.role == "system":
                return msg.content
        return ""

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    cached: bool = False
    cache_type: Optional[str] = None  # "exact" or "semantic"
    latency_ms: Optional[float] = None

class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
