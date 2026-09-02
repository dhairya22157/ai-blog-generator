import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import END, START, StateGraph


load_dotenv()


class BlogState(TypedDict, total=False):
    topic: str
    needs_research: bool
    research: str
    content: str
    needs_image: bool
    image_prompt: str
    image_path: str


def research_decision(state: BlogState) -> dict[str, bool]:
    topic = state["topic"]
    return {"needs_research": "latest" in topic.lower() or "current" in topic.lower()}


def route_research(state: BlogState) -> str:
    return "research" if state.get("needs_research", False) else "generate-content"


def research_node(state: BlogState) -> dict[str, str]:
    return {"research": f"Research summary for: {state['topic']}"}


def get_chat_model(temperature: float = 0.7) -> ChatHuggingFace:
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN not found. Add it to your .env file."
        )

    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="conversational",
        temperature=temperature,
        max_new_tokens=512,
        huggingfacehub_api_token=api_token,
    )
    return ChatHuggingFace(llm=endpoint)


def generate_content(state: BlogState) -> dict[str, str]:
    topic = state["topic"]
    research = state.get("research", "")
    prompt = f"Write a clear blog post about '{topic}'."
    if research:
        prompt += f" Use this research context:\n{research}"
    response = get_chat_model(temperature=0.7).invoke(prompt)
    return {"content": getattr(response, "content", str(response))}


def parse_json_object(raw_response: str) -> dict[str, object]:
    text = raw_response.strip()

    # 1. Clean markdown code fences if present (e.g., ```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # 2. Try direct JSON parsing
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 3. Try regex match for outermost { ... }
    match = re.search(r"\{[\s\S]*\}", raw_response)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 4. Fallback: Parse conversational or unstructured response
    lower = raw_response.lower()
    needs_image = False

    positive_signals = [
        "does need", "needs an image", "should include an image", "would benefit from an image",
        "need an image", "needs_image: true", '"needs_image": true', "'needs_image': true", "true", "yes"
    ]

    if any(pos in lower for pos in positive_signals) and not any(neg in lower for neg in ["does not need", "doesn't need", "no image"]):
        needs_image = True

    # Try extracting image prompt
    prompt_match = re.search(r'(?:image_prompt|image prompt|prompt|illustration)[^:\n]*[:=]\s*["\']?([^"\'\n\r]+)["\']?', raw_response, re.IGNORECASE)
    image_prompt = ""
    if prompt_match:
        image_prompt = prompt_match.group(1).strip()

    return {"needs_image": needs_image, "image_prompt": image_prompt}


def image_decision(state: BlogState) -> dict[str, object]:
    topic = state.get("topic", "")
    content = state.get("content", "")

    messages = [
        SystemMessage(
            content=(
                "You are an AI assistant that determines if a blog post needs an accompanying image.\n"
                "You must respond ONLY with a single valid JSON object and nothing else (no introductory text, no markdown backticks, no explanation).\n"
                'JSON format: {"needs_image": true, "image_prompt": "detailed image generation prompt"}'
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {topic}\n\n"
                f"Blog post:\n{content}\n\n"
                "Does this blog post need an image? Return only JSON:"
            )
        ),
    ]

    response = get_chat_model(temperature=0.1).invoke(messages)
    raw_response = getattr(response, "content", str(response)).strip()
    decision = parse_json_object(raw_response)

    needs_image = bool(decision.get("needs_image", False))
    image_prompt = str(decision.get("image_prompt", "") or "").strip()

    if needs_image and not image_prompt:
        image_prompt = f"A detailed diagram and visual illustration explaining {topic}, high quality digital art"

    return {"needs_image": needs_image, "image_prompt": image_prompt}


def route_image(state: BlogState) -> str:
    return "generate-image" if state.get("needs_image", False) else END


def generate_image_from_api(prompt: str, output_path: str = "blog_picture.png") -> str:
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found. Add it to your .env file.")
    image = InferenceClient(token=api_token).text_to_image(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell",
    )
    if hasattr(image, "save"):
        image.save(output_path)
    elif isinstance(image, (bytes, bytearray)):
        with open(output_path, "wb") as f:
            f.write(image)
    return output_path


def generate_image(state: BlogState) -> dict[str, str]:
    return {"image_path": generate_image_from_api(state["image_prompt"])}


def insert_image(state: BlogState) -> dict[str, str]:
    return {"content": f"{state['content']}\n\n![Blog Image]({state['image_path']})\n"}


def build_blog_graph():
    graph = StateGraph(BlogState)
    graph.add_node("research_decision", research_decision)
    graph.add_node("research", research_node)
    graph.add_node("generate-content", generate_content)
    graph.add_node("image_decision", image_decision)
    graph.add_node("generate-image", generate_image)
    graph.add_node("insert_image", insert_image)
    graph.add_edge(START, "research_decision")
    graph.add_conditional_edges("research_decision", route_research, {
        "research": "research", "generate-content": "generate-content"
    })
    graph.add_edge("research", "generate-content")
    graph.add_edge("generate-content", "image_decision")
    graph.add_conditional_edges("image_decision", route_image, {
        "generate-image": "generate-image", END: END
    })
    graph.add_edge("generate-image", "insert_image")
    graph.add_edge("insert_image", END)
    return graph.compile()


def generate_blog(topic: str) -> BlogState:
    return build_blog_graph().invoke({"topic": topic})


if __name__ == "__main__":
    result = generate_blog(
        "Explain the self-attention mechanism in deep learning, including "
        "how it works, why it is important, and a simple real-world example"
    )
    print(result["content"])
