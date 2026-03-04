from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import asyncio
import os
import signal
import sys
from typing import List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


class FirecrawlAgent:
    """Firecrawl MCP Agent using Ollama GPT-OSS 20B for web scraping."""

    MODEL_NAME = "gpt-oss:20b"

    def __init__(self) -> None:
        self.model = ChatOllama(
            model=self.MODEL_NAME,
            temperature=0.1,
            num_ctx=8192  # Larger context for tool usage
        )
        self.server_params = StdioServerParameters(
            command="npx",
            env={
                "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
            },
            args=["firecrawl-mcp"]
        )
        self.session: ClientSession | None = None
        self.agent: Any | None = None
        self.messages: List[dict] = []
        self.running = True

    def setup_system_message(self) -> None:
        """Initialize system message optimized for Ollama."""
        self.messages = [{
            "role": "system",
            "content": (
                "You are an expert web research assistant using Firecrawl MCP tools. "
                "You can scrape websites, crawl pages, search the web, and extract structured data. "
                "Always think step-by-step before using tools. Use the most appropriate tool for each task. "
                "Provide clear, actionable insights from the scraped data."
            )
        }]

    async def initialize_session(self) -> None:
        """Initialize MCP session and load Firecrawl tools."""
        logger.info("Initializing Firecrawl MCP session...")

        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
                tools = await load_mcp_tools(session)
                self.agent = create_react_agent(self.model, tools)

                tool_names = ", ".join(tool.name for tool in tools)
                print(f"Available Tools: {tool_names}")
                print("-" * 60)
                logger.info(f"Loaded {len(tools)} Firecrawl tools")

    async def process_query(self, user_input: str) -> None:
        """Process user query with conversation history."""
        # Truncate very long inputs
        if len(user_input) > 10000:
            user_input = user_input[:10000] + "\n[Input truncated]"

        self.messages.append({"role": "user", "content": user_input})

        try:
            logger.info(f"Processing query with {len(self.messages)} message history")
            result = await self.agent.ainvoke({"messages": self.messages})

            # Extract latest assistant message
            ai_message = result["messages"][-1].content
            print(f"\nAgent: {ai_message}")

            # Maintain conversation history (keep last 10 exchanges)
            self.messages.append({"role": "assistant", "content": ai_message})
            if len(self.messages) > 21:  # system + 10 exchanges
                self.messages = [self.messages[0]] + self.messages[-20:]

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg)
            print(f"Error: {error_msg}")

    def register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""

        def shutdown_handler(signum: int, frame: Any) -> None:
            logger.info("Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

    async def run(self) -> None:
        """Main interactive loop."""
        print("Firecrawl MCP Agent (Ollama GPT-OSS 20B)")
        print("Commands: 'quit', 'exit', 'q', 'clear', or Ctrl+C")
        print("-" * 50)

        self.setup_system_message()
        self.register_signal_handlers()

        try:
            await self.initialize_session()

            while self.running:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, input, "You: "
                    )
                    user_input = user_input.strip()

                    if user_input.lower() in {"quit", "exit", "q"}:
                        logger.info("User requested exit")
                        break

                    if user_input.lower() == "clear":
                        self.messages = [self.messages[0]]  # Keep system message
                        print("\nConversation history cleared.")
                        continue

                    if user_input:
                        await self.process_query(user_input)

                except EOFError:
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"Critical error: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """Perform cleanup on shutdown."""
        logger.info("Cleaning up resources...")
        self.session = None
        self.agent = None
        self.messages = []
        print("\nGoodbye!")


async def main() -> None:
    """Main entry point."""
    agent = FirecrawlAgent()
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
