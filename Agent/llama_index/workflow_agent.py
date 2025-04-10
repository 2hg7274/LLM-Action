import os

import time
from typing import Any
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import (
    Event,
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step
)

from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import ActionReasoningStep, ObservationReasoningStep
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool

from configs import MODEL_PATH
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.tools import FunctionTool
from tools import (
    get_top_processes_by_memory, 
    kill_process,
    get_cpu_usage,
    get_disk_usage,
    get_memory_info,
    get_system_error_logs,
    restart_service
)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


## ========== Event ==========
class PrepEvent(Event):
    """
    다음 단계로 진행하기 위해 준비가 완료 되었음을 알리는 이벤트
    """
    pass

class InputEvent(Event):
    """
    LLM에게 전달할 입력(대화 내역 등)을 포함하는 이벤트
    """
    input: list[ChatMessage]

class StreamEvent(Event):
    """
    LLM의 스트리밍 응답을 전달할 때 사용
    """
    delta: str

class ToolCallEvent(Event):
    """
    LLM의 출력에 따라 tool를 호출할 때 발생하며, 호출할 도구 목록 및 인자를 포함
    """
    tool_calls: list[ToolSelection]

class FunctionOutputEvent(Event):
    """
    도구 호출 후 도구의 출력 결과를 나타내는 이벤트
    """
    output: ToolOutput


# ========== Workflow ==========
class ReActAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        extra_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm
        self.formatter = ReActChatFormatter.from_context(
            context=extra_context or ""
        )
        self.output_parser = ReActOutputParser()

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        """
        역할
            - 대화의 시작점
            - 컨텍스트에서 기존 sources (툴의 결과 등)를 초기화
            - 사용자 입력을 받아 ChatMemoryBuffer에 저장하여 대화 기록(메모리)를 갱신
            - 현재 추론(current_reasoning) 상태를 초기화
        
        입력 및 출력
            - 입력으로 StartEvent가 주어지면, 사용자 메시지를 메모리에 기록 후 PrepEvent를 반환하여 다음 단계로 넘어감
        """
        # clear sources
        await ctx.set("sources", [])

        # init memory if needed
        memory = await ctx.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # clear current reasoning
        await ctx.set("current_reasoning", [])

        # set memory
        await ctx.set("memory", memory)

        return PrepEvent()

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: PrepEvent
    ) -> InputEvent:
        """
        역할
            - 저장된 메모리(대화 내역)와 현재까지의 추론 내용을 불러옴
            - ReAct 방식에 맞춰 LLM에 전달할 프롬프트를 생성하기 위해 self.formatter를 사용
        
        출력
            - 생성된 프롬프트(ChatMessage)를 담은 InputEvent를 반환
        """
        # get chat history
        memory = await ctx.get("memory")
        chat_history = memory.get()
        current_reasoning = await ctx.get("current_reasoning", default=[])

        # format the prompt with react instructions
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        """
        역할
            - LLM의 astream_chat를 통해 스트리밍 응답을 받아 처리
            - 스트리밍 중에는 StreamEvent를 통해 부분 결과를 실시간 출력
            - 전체 응답이 도착하면, ReActOutputParser를 이용해 응답을 파싱.

        분기 처리
            - 최종 응답(is_done):
                파싱된 결과가 완료 상태라면, LLM이 생성한 답변을 메모리에 추가하고,
                StopEvent에 최종 응답 및 추가 정보(tool의 출력 결과와 추론 과정)를 담아 반환

            - 도구 호출 요청(ActionReasoningStep):
                LLM의 추론 결과가 도구 호출을 요구하는 경우, 
                해당 도구 이름과 인자를 포함한 ToolCallEvnet를 반환

            - 예외 처리
                만약 출력 파싱에서 예외가 발생하면, 에러 정보를 추론 과정(current_reasoning)에 추가한 후
                다시 PrepEvent를 반환하여 반복 처리를 시도
        """
        chat_history = ev.input
        current_reasoning = await ctx.get("current_reasoning", default=[])
        memory = await ctx.get("memory")

        response_gen = await self.llm.astream_chat(chat_history)
        async for response in response_gen:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            current_reasoning.append(reasoning_step)

            if reasoning_step.is_done:
                memory.put(
                    ChatMessage(
                        role="assistant", content=reasoning_step.response
                    )
                )
                await ctx.set("memory", memory)
                await ctx.set("current_reasoning", current_reasoning)

                sources = await ctx.get("sources", default=[])

                return StopEvent(
                    result={
                        "response": reasoning_step.response,
                        "sources": [sources],
                        "reasoning": current_reasoning,
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
        except Exception as e:
            current_reasoning.append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )
            await ctx.set("current_reasoning", current_reasoning)

        # if no tool calls or final response, iterate again
        return PrepEvent()

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> PrepEvent:
        """
        역할
            - LLM의 추론 단계 중 도구 호출이 필요한 경우, ToolCallEvent 내에 지정된 도구들을 실제로 호출
            - tools_by_name 딕셔너리를 사용하여 도구 이름에 따른 실제 도구 객체를 찾고,
              인자(tool_kwargs)를 전달하여 도구 함수를 실행.
            - 도구의 출력 결과는 sources 리스트에 저장되고, 추론 과정에도 그 결과를 Observation 형태로 추가
            - 만약 지정된 도구가 존재하지 않거나 호출 중 예외가 발생하면, 해당 에러 메시지를 추론 과정에 기록

        출력
            - 도구 호출 후 업데이트된 상태를 가지고 다시 PrepEvent를 반환하여,
              이후 LLM에게 추가 입력으로 보내거나 최종 응답 처리를 진행. 
        """
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        current_reasoning = await ctx.get("current_reasoning", default=[])
        sources = await ctx.get("sources", default=[])

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                current_reasoning.append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                continue

            try:
                start_time = time.time()
                tool_output = tool(**tool_call.tool_kwargs)
                end_time = time.time()
                duration = end_time - start_time

                sources.append(tool_output)
                current_reasoning.append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
                msg = f"(tool execution took {duration:.2f} seconds)"
                ctx.write_event_to_stream(StreamEvent(delta=f"\n[Tool execution time] {msg}\n"))
            except Exception as e:
                current_reasoning.append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )
        

        # save new state in context
        await ctx.set("sources", sources)
        await ctx.set("current_reasoning", current_reasoning)

        # prep the next iteraiton
        return PrepEvent()
    
    

# ========== tools ==========
tools = [
    FunctionTool.from_defaults(get_top_processes_by_memory),
    FunctionTool.from_defaults(kill_process),
    # FunctionTool.from_defaults(get_cpu_usage),
    # FunctionTool.from_defaults(get_disk_usage),
    # FunctionTool.from_defaults(get_memory_info),
    # FunctionTool.from_defaults(get_system_error_logs),
    # FunctionTool.from_defaults(restart_service)
]


# ========== Agent ==========
llm = HuggingFaceLLM(
        model_name=MODEL_PATH,
        tokenizer_name=MODEL_PATH,
        max_new_tokens=4096,
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        device_map="auto",
    )
agent = ReActAgent(
    llm=llm, 
    tools=tools, 
    timeout=120, 
    verbose=False, 
    extra_context="""
    Here is a description of the server log data:
    - timestamp: The time at which the data was collected or predicted (ISO format string)
    - cpu_usage_percent: The CPU usage as a percentage
    - memory_usage_percent: The memory usage as a percentage
    - disk_usage_percent: The disk usage as a percentage
    - confidence: The confidence of the model in the prediction result (a value between 0 and 1, with values ​​closer to 1 indicating higher confidence)
    - error_occur: Whether the data at that point in time predicted an error (True or False)
    Given the log data, use tools to troubleshoot the error."""
)


# ========== main ==========
async def main():
    ctx = Context(agent)

    while True:
        user_input = input("USER: ")
        if user_input.strip().upper() == "END":
            break

        print()
        print("====="*10)
        start_time = time.time()  # 전체 처리 시작 시간

        handler = agent.run(input=user_input, ctx=ctx)

        async for event in handler.stream_events():
            if isinstance(event, StreamEvent):
                print(event.delta, end="", flush=True)

        response = await handler

        end_time = time.time()  # 전체 처리 종료 시간
        total_duration = end_time - start_time

        print()
        print("====="*10)
        print(f"\n\n⏱ Total processing time: {total_duration:.2f} seconds")

        print("\nAGENT:")
        print(response['response'])
        print("\n\n\n")


if __name__=="__main__":
   import asyncio
   asyncio.run(main())