#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_mineru_and_ingest.py

Pipeline:
PDF -> MinerU output dir -> ingest_mineru_qdrant.py

Optional:
NVIDIA Partner Hub Rubin PDFs -> nvidia_rubin_pdfs -> MinerU missing PDFs only

Example:
python run_mineru_and_ingest.py ^
  --pdf_root "D:\\docker_hub\\AI_Project\\nvidia_rubin_pdfs" ^
  --mineru_out_root "D:\\work\\mineru_out" ^
  --qdrant_url "http://127.0.0.1:6333" ^
  --collection "mineru_docs" ^
  --mineru_cmd "magic-pdf"
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

NVIDIA_DOCUMENT_URL = "https://partners.nvidia.com/DocumentDetails?siteID=309632"
DEFAULT_DOWNLOAD_DIR = Path(__file__).resolve().parents[1] / "pdf_new"


def default_session_dir() -> Path:
    if sys.platform.startswith("linux"):
        return Path.home() / ".cache" / "nvidia_partner_session"
    return Path(__file__).resolve().parents[1] / "nvidia_session"


def find_pdfs(pdf_root: Path):
    if not pdf_root.exists():
        return []
    return sorted(pdf_root.rglob("*.pdf"))


def expected_output_dir(pdf_path: Path, mineru_out_root: Path) -> Path:
    return mineru_out_root / pdf_path.stem


def expected_original_pdf(pdf_path: Path, mineru_out_root: Path) -> Path:
    stem = pdf_path.stem
    return mineru_out_root / stem / stem / "hybrid_auto" / f"{stem}_original.pdf"


def filesystem_path(path: Path) -> Path:
    if not sys.platform.startswith("win"):
        return path
    absolute = str(path.absolute())
    if absolute.startswith("\\\\?\\"):
        return path
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute.lstrip("\\"))
    return Path("\\\\?\\" + absolute)


def path_exists(path: Path) -> bool:
    if path.exists():
        return True
    try:
        return filesystem_path(path).exists()
    except OSError:
        return False


def path_rglob(root: Path, pattern: str):
    try:
        return list(filesystem_path(root).rglob(pattern))
    except OSError:
        return []


def is_mineru_output_ready(pdf_path: Path, mineru_out_root: Path) -> bool:
    expected_original = expected_original_pdf(pdf_path, mineru_out_root)
    if path_exists(expected_original):
        return True

    out_dir = expected_output_dir(pdf_path, mineru_out_root)
    md = path_rglob(out_dir, "*.md")
    has_images = any(path.name == "images" and path.is_dir() for path in path_rglob(out_dir, "images"))
    has_json = (
        bool(path_rglob(out_dir, "*content_list*.json"))
        or bool(path_rglob(out_dir, "*middle*.json"))
    )
    return bool(md) and has_images and has_json


def sanitize_filename(name: str) -> str:
    return "".join("_" if char in '<>:"/\\|?*' else char for char in name).strip() or "download.bin"


def is_pdf_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() == ".pdf"


def filename_from_response(url: str, headers: dict, index: int) -> str:
    disposition = headers.get("content-disposition") or headers.get("Content-Disposition") or ""
    for part in disposition.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            return sanitize_filename(part.split("=", 1)[1].strip().strip('"'))
    path_name = Path(urlparse(url).path).name
    if path_name:
        return sanitize_filename(path_name)
    return f"rubin_download_{index + 1}.pdf"


def safe_goto(page, url: str, playwright_error):
    try:
        page.goto(url, wait_until="commit", timeout=60000)
        page.wait_for_load_state("domcontentloaded", timeout=60000)
    except playwright_error as exc:
        if "interrupted by another navigation" not in str(exc):
            raise
        page.wait_for_load_state("domcontentloaded", timeout=60000)


def wait_network_idle(page, timeout_error, timeout: int = 15000):
    try:
        page.wait_for_load_state("networkidle", timeout=timeout)
    except timeout_error:
        pass


def find_search_box(page):
    selectors = [
        "input[placeholder='Enter keywords here or apply filters below']",
        "input[placeholder*='Enter keywords']",
        "input[placeholder*='keywords']",
        "#SearchText",
        "#txtSearch",
        "input[name*='search' i]",
        "input[id*='search' i]",
    ]
    for selector in selectors:
        locator = page.locator(selector)
        count = locator.count()
        for index in range(count):
            item = locator.nth(index)
            try:
                if item.is_visible():
                    return item
            except Exception:
                pass

    print("Search box not found. Current URL:", page.url)
    print("Inputs on page:")
    for index in range(page.locator("input").count()):
        item = page.locator("input").nth(index)
        try:
            print(
                index,
                "id=", item.get_attribute("id"),
                "name=", item.get_attribute("name"),
                "placeholder=", item.get_attribute("placeholder"),
                "visible=", item.is_visible(),
            )
        except Exception:
            pass
    raise RuntimeError("Could not find the visible Content Library search input.")


def click_search(page):
    selectors = [
        "button:has-text('SEARCH')",
        "input[value='SEARCH']",
        "a:has-text('SEARCH')",
        "text=SEARCH",
    ]
    for selector in selectors:
        locator = page.locator(selector)
        count = locator.count()
        for index in range(count):
            item = locator.nth(index)
            try:
                if item.is_visible():
                    item.click()
                    return
            except Exception:
                pass
    raise RuntimeError("Could not find a visible SEARCH button.")


def href_from_download_element(page, element):
    candidates = [
        "el => el.href || el.getAttribute('href')",
        "el => el.closest('a') ? el.closest('a').href : null",
        "el => el.parentElement && el.parentElement.closest('a') ? el.parentElement.closest('a').href : null",
    ]
    for script in candidates:
        try:
            value = element.evaluate(script)
            if value:
                return urljoin(page.url, value)
        except Exception:
            pass
    return None


def save_response_download(context, url: str, index: int, download_dir: Path):
    response = context.request.get(url, timeout=60000)
    if not response.ok:
        raise RuntimeError(f"Download request failed: {response.status} {response.status_text}")
    filename = filename_from_response(url, response.headers, index)
    if not is_pdf_filename(filename):
        print(f"[DOWNLOAD][SKIP] non-PDF file: {filename}")
        return
    target = download_dir / filename
    if path_exists(target):
        print(f"[DOWNLOAD][SKIP] existing file: {target}")
        return
    target.write_bytes(response.body())
    print(f"[DOWNLOAD] saved via request: {target}")


def save_browser_download(download, download_dir: Path):
    filename = sanitize_filename(download.suggested_filename)
    if not is_pdf_filename(filename):
        print(f"[DOWNLOAD][SKIP] non-PDF file: {filename}")
        return
    target = download_dir / filename
    if path_exists(target):
        print(f"[DOWNLOAD][SKIP] existing file: {target}")
        return
    download.save_as(target)
    print(f"[DOWNLOAD] saved: {target}")


def visible_locator(page, selectors):
    for selector in selectors:
        locator = page.locator(selector)
        count = locator.count()
        for index in range(count):
            item = locator.nth(index)
            try:
                if item.is_visible():
                    return item
            except Exception:
                pass
    return None


def accept_eula_and_download(page, timeout_error, download_dir: Path) -> bool:
    accept = visible_locator(
        page,
        [
            "button:has-text('ACCEPT AND DOWNLOAD EULA')",
            "input[value='ACCEPT AND DOWNLOAD EULA']",
            "text=ACCEPT AND DOWNLOAD EULA",
            "button:has-text('ACCEPT')",
            "input[value='ACCEPT']",
            "text=ACCEPT",
        ],
    )
    if accept is None:
        return False
    try:
        with page.expect_download(timeout=30000) as download_info:
            accept.click()
        save_browser_download(download_info.value, download_dir)
        return True
    except timeout_error:
        print("[DOWNLOAD] accepted EULA, but no browser download event was emitted.")
        return True


def restore_results_page(page, results_url: str, playwright_error, timeout_error):
    if results_url and page.url != results_url:
        safe_goto(page, results_url, playwright_error)
        wait_network_idle(page, timeout_error)


def is_login_page(page) -> bool:
    if "/Login" in page.url:
        return True
    username = page.locator("#Username, input[name='Username']").first
    password = page.locator("#Password, input[name='Password']").first
    try:
        return username.is_visible() and password.is_visible()
    except Exception:
        return False


def wait_until_not_login_page(page, timeout_error, timeout_ms: int = 120000) -> bool:
    try:
        page.wait_for_function(
            """() => {
                const url = window.location.href;
                const username = document.querySelector("#Username, input[name='Username']");
                const password = document.querySelector("#Password, input[name='Password']");
                const loginInputsVisible = username && password &&
                    username.offsetParent !== null && password.offsetParent !== null;
                return !url.includes("/Login") && !loginInputsVisible;
            }""",
            timeout=timeout_ms,
        )
        return True
    except timeout_error:
        return not is_login_page(page)


def ensure_playwright_available():
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Playwright is not installed in the Python interpreter running this script. "
            "Activate the venv that has Playwright installed, or run: "
            "python3 -m pip install playwright && python3 -m playwright install chromium"
        ) from exc
    return PlaywrightError, PlaywrightTimeoutError, sync_playwright


def ensure_visible_linux_display(browser_headless: bool):
    if browser_headless or not sys.platform.startswith("linux"):
        return
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "No Linux X display is available for headed Playwright. "
            "Run from a visible WSL/Linux desktop with DISPLAY set, or use --browser_headless "
            "with an already logged-in --nvidia_session_dir. xvfb-run hides the login window, "
            "so it cannot be used for manual login."
        )

def is_software_package(link) -> bool:
    """从 Download 元素向上查找当前结果条目的 Content Type。"""
    container = link

    for _ in range(12):
        container = container.locator("xpath=..")
        if container.count() == 0:
            return False

        # 避免上溯到整个结果列表，把其他条目的类型误认为当前类型。
        if container.locator("text=Download").count() > 1:
            return False

        text = container.inner_text(timeout=5000)
        text = re.sub(r"\s+", " ", text).strip()

        match = re.search(
            r"\bContent\s*Type\s*:?\s*Software\s+Package\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return True

    return False

def check_visible_content_type(page) -> str:
    """返回 software_package / other / unknown。"""
    labels = page.get_by_text(
        re.compile(r"^\s*Content\s*Type\s*[:：]?\s*$", re.I)
    )

    for index in range(labels.count()):
        label = labels.nth(index)
        if not label.is_visible():
            continue

        container = label

        for _ in range(6):
            container = container.locator("xpath=..")
            if container.count() == 0:
                break

            text = re.sub(
                r"\s+", " ", container.inner_text()
            ).strip()

            # 只接受能明确识别为单个字段的文本。
            match = re.fullmatch(
                r"Content\s*Type\s*[:：]?\s+(.+)",
                text,
                flags=re.I,
            )
            if not match:
                continue

            value = match.group(1).strip()

            if value.casefold() == "software package":
                return "software_package"

            # 上溯到了整个详情区域，不能把后续字段当作类型值。
            if re.search(
                r"\bContent\s+(Class|Properties|Expiry)\b"
                r"|\bPosted\s+Date\b",
                value,
                flags=re.I,
            ):
                continue

            print(f"[DOWNLOAD] Content Type: {value}")
            return "other"

    return "unknown"

def download_rubin_pdfs(
    download_dir: Path,
    user_data_dir: Path,
    url: str,
    search_text: str,
    skip_login_prompt: bool,
    browser_headless: bool,
):
    PlaywrightError, PlaywrightTimeoutError, sync_playwright = ensure_playwright_available()
    ensure_visible_linux_display(browser_headless)

    download_dir.mkdir(parents=True, exist_ok=True)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=browser_headless,
            accept_downloads=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        page = context.pages[0] if context.pages else context.new_page()

        safe_goto(page, url, PlaywrightError)

        if not skip_login_prompt:
            for attempt in range(3):
                input("如果还没登录，请在浏览器里手动登录；登录完成并进入页面后按 Enter...")
                if "partners.nvidia.com" not in page.url:
                    safe_goto(page, url, PlaywrightError)
                wait_network_idle(page, PlaywrightTimeoutError)
                if wait_until_not_login_page(page, PlaywrightTimeoutError):
                    break
                if attempt < 2:
                    print(f"[LOGIN] 现在仍在 NVIDIA 登录页: {page.url}")
                    print("[LOGIN] 请确认浏览器里已经登录成功并跳转离开 Login 页面，然后再按 Enter。")
        else:
            if "partners.nvidia.com" not in page.url:
                safe_goto(page, url, PlaywrightError)
            wait_network_idle(page, PlaywrightTimeoutError)

        if is_login_page(page):
            context.close()
            raise RuntimeError(
                "Still on NVIDIA login page. Complete the login in the Chromium window before pressing "
                "Enter. If no Chromium window is visible, run without xvfb-run/headless on a visible "
                "desktop, or pass --nvidia_session_dir to a session directory that already contains a "
                "logged-in NVIDIA Partner Hub browser profile."
            )

        search = find_search_box(page)
        search.fill(search_text)
        click_search(page)
        wait_network_idle(page, PlaywrightTimeoutError)
        page.wait_for_timeout(3000)

        results_url = page.url
        download_links = page.locator("text=Download")
        count = download_links.count()
        print(f"[DOWNLOAD] found {count} download link(s)")

        # 临时诊断：不发起下载。
        input(
            "请在浏览器中打开一个 Software Package 的 "
            "Document Details 弹窗，不要点击弹窗里的 Download；"
            "打开后回到这里按 Enter..."
        )

        print("[DEBUG] URL:", page.url)

        labels = page.get_by_text(
            re.compile(r"^\s*Content\s*Type\s*[:：]?\s*$", re.I)
        )
        print("[DEBUG] Content Type 标签数量:", labels.count())

        for i in range(labels.count()):
            label = labels.nth(i)
            if not label.is_visible():
                continue

            node = label
            for level in range(4):
                print(f"\n[DEBUG] Content Type ancestor level={level}")
                print(node.evaluate("el => el.outerHTML").strip()[:12000])

                node = node.locator("xpath=..")
                if node.count() == 0:
                    break

        # 查看详情中的下载控件，但不点击。
        links = page.get_by_text("Download", exact=True)
        for i in range(links.count()):
            link = links.nth(i)
            if link.is_visible():
                print("\n[DEBUG] visible Download:")
                print(
                    link.evaluate(
                        "el => el.parentElement "
                        "? el.parentElement.outerHTML : el.outerHTML"
                    )[:6000]
                )

        input("诊断完成，按 Enter 关闭浏览器...")
        context.close()
        return

        context.close()

def resolve_mineru_command(mineru_cmd: str) -> list[str]:
    candidates = [mineru_cmd]
    if mineru_cmd == "mineru":
        candidates.insert(0, "magic-pdf")
    elif mineru_cmd == "magic-pdf":
        candidates.append("mineru")

    for candidate in dict.fromkeys(candidates):
        if Path(candidate).exists():
            return [candidate]
        resolved = shutil.which(candidate)
        if resolved:
            return [resolved]

    module_candidates = ["magic_pdf", "mineru"]
    for module_name in module_candidates:
        if importlib.util.find_spec(module_name):
            return [sys.executable, "-m", module_name]

    raise FileNotFoundError(
        "MinerU command was not found. Install MinerU in the same Python/PyCharm environment "
        "or pass the full executable path with --mineru_cmd. Tried: "
        + ", ".join(dict.fromkeys(candidates + [f"python -m {name}" for name in module_candidates]))
    )


def run_mineru(pdf_path: Path, out_dir: Path, mineru_cmd: list[str]):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 这里给一个通用占位命令模板
    # 你需要按你的 MinerU 实际 CLI 调整
    cmd = mineru_cmd + [
        "-p",
        str(pdf_path),
        "-o",
        str(out_dir),
    ]

    print(f"[MinerU] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def run_ingest(out_dir: Path, qdrant_url: str, collection: str, ingest_script: Path, local_llm_url: str = ""):
    cmd = [
        sys.executable,
        str(ingest_script),
        "--mineru_dir",
        str(out_dir),
        "--qdrant_url",
        qdrant_url,
        "--collection",
        collection,
    ]
    if local_llm_url:
        cmd.extend(["--local_llm_url", local_llm_url])

    print(f"[INGEST] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_root", default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--mineru_out_root", required=True)
    parser.add_argument("--qdrant_url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="mineru_docs")
    parser.add_argument("--mineru_cmd", default="magic-pdf")
    parser.add_argument("--local_llm_url", default="")
    parser.add_argument(
        "--ingest_script",
        default=str(Path(__file__).with_name("ingest_pdf_with_mineru_qdrant_full.py")),
    )
    parser.add_argument("--force_mineru", action="store_true")
    parser.add_argument("--force_ingest", action="store_true")
    parser.add_argument("--download_rubin", action="store_true")
    parser.add_argument("--download_only",action="store_true",help="只下载文档，不执行 MinerU 转换",)
    parser.add_argument("--download_url", default=NVIDIA_DOCUMENT_URL)
    parser.add_argument("--download_search", default="Rubin")
    parser.add_argument("--download_dir", default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--nvidia_session_dir", default=str(default_session_dir()))
    parser.add_argument("--skip_login_prompt", action="store_true")
    parser.add_argument("--browser_headless", action="store_true")
    args = parser.parse_args()

    pdf_root = Path(args.pdf_root)
    mineru_out_root = Path(args.mineru_out_root)
    ingest_script = Path(args.ingest_script)

    if args.download_rubin:
        download_dir = Path(args.download_dir)
        download_rubin_pdfs(
            download_dir=download_dir,
            user_data_dir=Path(args.nvidia_session_dir),
            url=args.download_url,
            search_text=args.download_search,
            skip_login_prompt=args.skip_login_prompt,
            browser_headless=args.browser_headless,
        )
        pdf_root = download_dir
        if args.download_only:
            print(f"[DONE] 下载流程结束，跳过 MinerU：{download_dir}")
            return

    pdfs = find_pdfs(pdf_root)
    print(f"[INFO] found {len(pdfs)} pdf(s) under {pdf_root}")

    pending_pdfs = []
    for pdf in pdfs:
        expected_original = expected_original_pdf(pdf, mineru_out_root)
        if not args.force_mineru and path_exists(expected_original):
            continue
        if args.force_mineru or not is_mineru_output_ready(pdf, mineru_out_root):
            pending_pdfs.append(pdf)

    mineru_cmd = None
    if pending_pdfs:
        try:
            mineru_cmd = resolve_mineru_command(args.mineru_cmd)
            print(f"[MinerU] resolved command: {' '.join(mineru_cmd)}")
        except Exception as e:
            print(f"[ERROR] {e}")
            return

    for pdf in pdfs:
        out_dir = expected_output_dir(pdf, mineru_out_root)
        expected_original = expected_original_pdf(pdf, mineru_out_root)
        if not args.force_mineru and path_exists(expected_original):
            print(f"[SKIP] MinerU original exists: {expected_original}")
            continue
        if args.force_mineru or not is_mineru_output_ready(pdf, mineru_out_root):
            try:
                print(f"[MinerU] missing original: {expected_original}")
                run_mineru(pdf, out_dir, mineru_cmd)
            except Exception as e:
                print(f"[ERROR] MinerU failed for {pdf}: {e}")
                continue
        else:
            print(f"[SKIP] MinerU output exists: {out_dir}")

if __name__ == "__main__":
    main()