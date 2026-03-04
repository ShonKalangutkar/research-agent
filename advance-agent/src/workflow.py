from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from .models import ResearchState, CompanyInfo, CompanyAnalysis
from .firecrawl import FirecrawlService
from .prompts import DeveloperToolsPrompts


class Workflow:
    def __init__(self):
        self.firecrawl = FirecrawlService()
        self.llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.1,
            num_ctx=8192
        )
        self.prompts = DeveloperToolsPrompts()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(ResearchState)
        graph.add_node("extract_tools", self._extract_tools_step)
        graph.add_node("research", self._research_step)
        graph.add_node("analyze", self._analyze_step)
        graph.set_entry_point("extract_tools")
        graph.add_edge("extract_tools", "research")
        graph.add_edge("research", "analyze")
        graph.add_edge("analyze", END)
        return graph.compile()

    def _extract_tools_step(self, state: ResearchState) -> Dict[str, Any]:
        print(f"🔍 Finding articles about: {state.query}")

        article_query = f"{state.query} tools comparison best alternatives"
        search_results = self.firecrawl.search_companies(article_query, num_results=3)

        all_content = ""
        for result in search_results:
            # Assume tuple: (title, url, ...) or (title, url, metadata...)
            if isinstance(result, tuple):
                url = result[1] if len(result) > 1 else None
            elif isinstance(result, dict):
                url = result.get("url") or result.get("metadata", {}).get("url")
            else:
                url = None

            if url:
                try:
                    scraped = self.firecrawl.scrape_company_pages(url)
                    if scraped and getattr(scraped, "markdown", None):
                        all_content += scraped.markdown[:1500] + "\n\n"
                except Exception as e:
                    print(f"Failed to scrape {url}: {e}")

        messages = [
            SystemMessage(content=self.prompts.TOOL_EXTRACTION_SYSTEM),
            HumanMessage(content=self.prompts.tool_extraction_user(state.query, all_content))
        ]

        try:
            response = self.llm.invoke(messages)
            tool_names = [
                name.strip()
                for name in response.content.strip().split("\n")
                if name.strip()
            ]
            print(f"Extracted tools: {', '.join(tool_names[:5])}")
            return {"extracted_tools": tool_names}
        except Exception as e:
            print(e)
            return {"extracted_tools": []}

    def _analyze_company_content(self, company_name: str, content: str) -> CompanyAnalysis:
        structured_llm = self.llm.with_structured_output(CompanyAnalysis)

        messages = [
            SystemMessage(content=self.prompts.TOOL_ANALYSIS_SYSTEM),
            HumanMessage(content=self.prompts.tool_analysis_user(company_name, content))
        ]

        try:
            analysis = structured_llm.invoke(messages)
            return analysis
        except Exception as e:
            print(e)
            return CompanyAnalysis(
                pricing_model="Unknown",
                is_open_source=None,
                tech_stack=[],
                description="Failed to analyze",
                api_available=None,
                language_support=[],
                integration_capabilities=[],
            )

    def _research_step(self, state: ResearchState) -> Dict[str, Any]:
        extracted_tools: List[str] = getattr(state, "extracted_tools", [])

        if not extracted_tools:
            print("⚠️ No extracted tools found, falling back to direct search")
            fallback_results = self.firecrawl.search_companies(
                state.query, num_results=4
            )

            # Assume search_companies returns List[Dict]
            tool_names = [
                r.get("metadata", {}).get("title", "Unknown")
                for r in fallback_results
            ]
        else:
            tool_names = extracted_tools[:4]

        print(f"🔬 Researching specific tools: {', '.join(tool_names)}")

        companies = []
        for tool_name in tool_names:
            search_results = self.firecrawl.search_companies(
                f"{tool_name} official site", num_results=1
            )

            if not search_results:
                continue

            # Assume search_results is List[Dict]
            first = search_results[0]
            url = first.get("url", "") or first.get("metadata", {}).get("url", "")

            company = CompanyInfo(
                name=tool_name,
                description=first.get("markdown", "No description"),
                website=url,
                tech_stack=[],
                competitors=[]
            )

            if url:
                try:
                    scraped = self.firecrawl.scrape_company_pages(url)
                    if scraped and getattr(scraped, "markdown", None):
                        content = scraped.markdown
                        analysis = self._analyze_company_content(company.name, content)

                        company.pricing_model = analysis.pricing_model
                        company.is_open_source = analysis.is_open_source
                        company.tech_stack = analysis.tech_stack
                        company.description = analysis.description
                        company.api_available = analysis.api_available
                        company.language_support = analysis.language_support
                        company.integration_capabilities = analysis.integration_capabilities
                except Exception as e:
                    print(f"Failed to analyze {tool_name}: {e}")

            companies.append(company)

        return {"companies": companies}

    def _analyze_step(self, state: ResearchState) -> Dict[str, Any]:
        print("Generating recommendations")

        company_data = ", ".join([
            company.json() for company in state.companies
        ])

        messages = [
            SystemMessage(content=self.prompts.RECOMMENDATIONS_SYSTEM),
            HumanMessage(content=self.prompts.recommendations_user(state.query, company_data))
        ]

        response = self.llm.invoke(messages)
        return {"analysis": response.content}

    def run(self, query: str) -> ResearchState:
        initial_state = ResearchState(query=query)
        final_state = self.workflow.invoke(initial_state)
        return ResearchState(**final_state)
