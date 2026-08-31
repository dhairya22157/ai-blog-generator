import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import END, START, StateGraph

load_dotenv()


class BlogState(TypedDict):
    topic: str
    needs_research: bool
    research: str
    content: str


def research_decision(state: BlogState):
    topic = state["topic"]
    if "latest" in topic.lower() or "current" in topic.lower():
        return {"needs_research": True}
    return {"needs_research": False}


def route_research(state: BlogState):
    if state.get("needs_research"):
        return "research"
    return "generate-content"


def research_node(state: BlogState):
    topic = state["topic"]
    return {"research": f"Research summary for: {topic}"}


def generate_content(state: BlogState) -> BlogState:
    topic = state["topic"]
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not api_token:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN not found. Add it to your .env file before running this script."
        )

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        temperature=0.7,
        max_new_tokens=512,
        huggingfacehub_api_token=api_token,
    )

    chat_model = ChatHuggingFace(llm=llm)
    response = chat_model.invoke(f"Write a blog post about '{topic}'.")
    content = getattr(response, "content", str(response))

    return {
        "topic": topic,
        "content": content,
    }


def build_blog_graph():
    graph = StateGraph(BlogState)
    graph.add_node("research_decision", research_decision)
    graph.add_node("research", research_node)
    graph.add_node("generate-content", generate_content)

    graph.add_edge(START, "research_decision")
    graph.add_conditional_edges(
        "research_decision",
        route_research,
        {
            "research": "research",
            "generate-content": "generate-content",
        },
    )
    graph.add_edge("research", "generate-content")
    graph.add_edge("generate-content", END)

    return graph.compile()


def generate_blog(topic: str):
    blog = build_blog_graph()
    return blog.invoke({"topic": topic})


if __name__ == "__main__":
    topic = "AI in healthcare"
    result = generate_blog(topic)
    print(result["content"])
