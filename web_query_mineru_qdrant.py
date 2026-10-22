# query_mineru_qdrant_web.py
from pathlib import Path
import streamlit as st
from query_mineru_qdrant import COLLECTION_NAME
#from FASTAPIModule import *
import requests

st.set_page_config(page_title="MinerU Qdrant QA", layout="wide")
API_URL = "http://127.0.0.1:28081/chat"

def render_source_card(i: int, d: dict):
    meta = d.get("metadata", {})
    text = d.get("text", "") or ""
    score = d.get("score", None)
    rerank_score = d.get("rerank_score", None)

    with st.container(border=True):
        st.markdown(f"### [{i}]")
        c1, c2, c3 = st.columns(3)
        c1.write(f"**type:** {meta.get('type')}")
        c2.write(f"**page:** {meta.get('page')}")
        c3.write(f"**score:** {score}")

        if rerank_score is not None:
            st.write(f"**rerank_score:** {rerank_score}")

        if meta.get("source_pdf"):
            st.write(f"**source_pdf:** {meta.get('source_pdf')}")

        if meta.get("image_path"):
            img_path = Path(meta["image_path"])
            st.write(f"**image_path:** {img_path}")

            if img_path.exists():
                try:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
                except Exception as e:
                    st.warning(f"图片显示失败: {e}")
            else:
                st.warning("图片文件不存在")

        if text.strip():
            st.markdown("**Text:**")
            st.write(text[:4000])


def main():
    st.title("MinerU Qdrant QA")
    st.caption("查询本地 Qdrant 中的文档结果，并展示文本 + 图片证据")

    with st.sidebar:
        st.header("Settings")
        qdrant_url = st.text_input("Qdrant URL", "http://127.0.0.1:6333")
        collection = st.text_input("Collection", COLLECTION_NAME)
        st.markdown("---")
        st.markdown("**说明**")
        st.write("如果 source 里有 `type=image` 且 `image_path` 有效，会直接显示图片。")

    question = st.text_area(
        "输入问题",
        height=120,
        placeholder="例如：Vera Rubin NVL72 architecture 是什么？"
    )

    run = st.button("查询", type="primary")

    if run:
        if not question.strip():
            st.warning("请输入问题")
            return
        with st.spinner("检索与生成中..."):
            #result = ask(question.strip(), qdrant_url, collection, req=req, pg=pg, mem=mem)
            result = requests.post(API_URL, json={
                "user_id": "HuangWei",
                "message": question.strip(),
                "qdrant_url": qdrant_url,
                "collection": collection}).json()

        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander("Prompt / Context", expanded=False):
            st.code(result["prompt"], language="text")

        st.subheader("Sources")
        docs = result.get("docs", [])
        if not docs:
            st.info("没有检索到结果")
        else:
            for i, d in enumerate(docs, 1):
                render_source_card(i, d)


if __name__ == "__main__":
    main()