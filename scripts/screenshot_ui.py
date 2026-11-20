"""Drive the chat UI in headless Chromium and take screenshots (desktop + phone)."""
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8501"
MESSAGES = sys.argv[2:] or ["ist 7 gerade?", "wer bist du?"]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
    page.goto(URL)
    page.wait_for_selector(".pill.ok", timeout=30000)
    for i, msg in enumerate(MESSAGES):
        page.fill("#inp", msg)
        page.click("#send")
        t0 = time.time()
        # wait until the answer bubble is finished (typing class removed)
        page.wait_for_function("document.querySelectorAll('.msg.bot.typing').length === 0 && "
                               f"document.querySelectorAll('.msg.bot').length >= {i + 2}", timeout=600000)
        print(f"{msg!r} answered in {time.time() - t0:.0f}s", flush=True)
    page.screenshot(path="docs/ui_desktop.png", full_page=True)
    answers = page.eval_on_selector_all(".msg.bot", "els => els.map(e => e.innerText)")
    print("bot bubbles:", answers)
    ctx = browser.new_context(**p.devices["iPhone 13"])
    m = ctx.new_page()
    m.goto(URL)
    m.wait_for_selector(".pill.ok", timeout=30000)
    m.screenshot(path="docs/ui_phone.png", full_page=False)
    browser.close()
