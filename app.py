import base64
import os
import re
import streamlit as st
from dotenv import load_dotenv

from blog import build_blog_graph


def render_markdown_with_images(content: str) -> str:
    """Converts local markdown image links to base64 Data URIs so Streamlit can render them inline."""
    def replace_image_path(match):
        alt = match.group(1)
        path = match.group(2)
        if os.path.exists(path) and not path.startswith("http") and not path.startswith("data:"):
            ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"![{alt}](data:image/{ext};base64,{b64})"
        return match.group(0)

    return re.sub(r"!\[(.*?)\]\((.*?)\)", replace_image_path, content)


# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="AI Agentic Blog Generator",
    page_icon="✍️",
    layout="wide",
)

st.title("✍️ Agentic AI Blog Generator")
st.markdown(
    "Generate comprehensive blog posts with automatic research routing and AI-generated visuals powered by **LangGraph**, **Llama-3.1**, and **FLUX.1**."
)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Configuration")
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    user_token = st.text_input(
        "Hugging Face API Token",
        value=api_token,
        type="password",
        help="Required for Llama-3.1 and FLUX inference.",
    )
    if user_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = user_token

    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        st.warning("⚠️ Please provide a Hugging Face API token to proceed.")

    st.divider()
    st.markdown("### 🔄 Graph Workflow")
    st.markdown(
        """
        1. **Research Decision**: Checks if topic needs current research.
        2. **Research Node**: Gathers context if needed.
        3. **Content Generation**: Generates full article via Llama-3.1.
        4. **Image Decision**: Decides if visual illustration is needed.
        5. **Image Generation**: Generates image via FLUX.1.
        6. **Publish**: Combines content & image.
        """
    )

# Pre-defined sample topics
sample_topics = [
    "Explain the self-attention mechanism in deep learning, including how it works, why it is important, and a simple real-world example",
    "Latest breakthroughs in quantum computing and their practical applications",
    "How to build autonomous agentic workflows using LangGraph and Python",
    "The architecture and future of multimodal generative AI models",
]

col1, col2 = st.columns([3, 1])
with col1:
    selected_sample = st.selectbox(
        "💡 Or pick an example topic:",
        ["-- Custom Topic --"] + sample_topics,
    )

with col2:
    st.write("")
    st.write("")
    clear_button = st.button("🧹 Clear Input", use_container_width=True)

if clear_button:
    default_topic = ""
elif selected_sample != "-- Custom Topic --":
    default_topic = selected_sample
else:
    default_topic = ""

topic = st.text_area(
    "Enter your blog topic / prompt:",
    value=default_topic,
    height=100,
    placeholder="e.g. Explain how transformers work in modern AI architectures...",
)

generate_btn = st.button("🚀 Generate Blog Post", type="primary", use_container_width=True)

if generate_btn:
    if not topic.strip():
        st.error("Please enter a topic before generating.")
    elif not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        st.error("Hugging Face API Token is required. Please add it to your `.env` or sidebar.")
    else:
        graph = build_blog_graph()
        
        with st.status("🤖 Running Agentic Blog Graph...", expanded=True) as status:
            state_accum = {"topic": topic.strip()}
            
            for event in graph.stream({"topic": topic.strip()}):
                for node_name, node_output in event.items():
                    state_accum.update(node_output)
                    
                    if node_name == "research_decision":
                        needs_res = node_output.get("needs_research", False)
                        status.write(f"🔍 **Research Decision**: {'Needs research context' if needs_res else 'Direct generation'}")
                    elif node_name == "research":
                        status.write("📚 **Research**: Gathered background information.")
                    elif node_name == "generate-content":
                        status.write("✍️ **Content Generator**: Blog post draft created.")
                    elif node_name == "image_decision":
                        needs_img = node_output.get("needs_image", False)
                        img_prompt = node_output.get("image_prompt", "")
                        status.write(
                            f"🎨 **Image Decision**: {'Image requested' if needs_img else 'No image needed'} "
                            f"{f'(`{img_prompt}`)' if img_prompt else ''}"
                        )
                    elif node_name == "generate-image":
                        status.write("🖼️ **Image Generator**: Generated image using FLUX.1.")
                    elif node_name == "insert_image":
                        status.write("📄 **Finalizing**: Embedded image into markdown.")

            status.update(label="✅ Blog Post Generation Complete!", state="complete", expanded=False)

        st.success("🎉 Blog post generated successfully!")
        
        # Display Results in Tabs
        tab1, tab2, tab3 = st.tabs(["📖 Rendered Blog", "📝 Markdown Source", "ℹ️ Generation Details"])
        
        with tab1:
            raw_content = state_accum.get("content", "")
            rendered_content = render_markdown_with_images(raw_content)
            st.markdown(rendered_content, unsafe_allow_html=True)

        with tab2:
            st.code(state_accum.get("content", ""), language="markdown")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="📥 Download Standard Markdown (.md)",
                    data=state_accum.get("content", ""),
                    file_name="generated_blog.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_d2:
                st.download_button(
                    label="📦 Download Self-Contained Markdown (Base64)",
                    data=render_markdown_with_images(state_accum.get("content", "")),
                    file_name="generated_blog_standalone.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        with tab3:
            st.json({
                "topic": state_accum.get("topic"),
                "needs_research": state_accum.get("needs_research"),
                "research": state_accum.get("research"),
                "needs_image": state_accum.get("needs_image"),
                "image_prompt": state_accum.get("image_prompt"),
                "image_path": state_accum.get("image_path"),
            })
