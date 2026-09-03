"""
generate_corpus.py
Builds a small (12-document) synthetic "intranet" corpus: PDFs mixing
extractable text with charts/diagrams, plus two standalone images. Several
documents are deliberately designed so the *answer* to a query lives only
in the visual content (a bar chart, a floor-plan label, a Gantt date) and
NOT in any text a PDF text-extractor would pull out -- this is what lets
the evaluation later show a genuine gap between text-only RAG and
multimodal retrieval, rather than a contrived one.

Run:
    python generate_corpus.py
Produces:
    data/pdfs/*.pdf, data/images/*.png, data/manifest.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

HERE = os.path.dirname(__file__)
PDF_DIR = os.path.join(HERE, "pdfs")
IMG_DIR = os.path.join(HERE, "images")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

manifest = []  # ground truth: doc_id, path, extractable_text, visual_only_facts


def add_manifest(doc_id, kind, path, text, visual_only_facts, image_paths=None):
    visual_facts = visual_only_facts
    manifest.append(
        {
            "doc_id": doc_id,
            "kind": kind,  # "pdf" or "image"
            "path": os.path.relpath(path, HERE),
            "extractable_text": text.strip(),
            "visual_only_facts": visual_facts,  # facts NOT present in `text`
            "image_paths": [os.path.relpath(p, HERE) for p in (image_paths or [])],
        }
    )


def make_chart_png(out_path, kind, **kw):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    if kind == "bar_revenue":
        regions = ["North", "South", "East", "West", "APAC"]
        values = [4.2, 3.1, 5.8, 2.9, 6.4]
        bars = ax.bar(regions, values, color="#3B6EA5")
        ax.set_title("Q3 Revenue by Region ($M)")
        ax.set_ylabel("Revenue ($M)")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.1, str(v), ha="center", fontsize=8)
    elif kind == "latency_line":
        import numpy as np
        t = np.arange(0, 60, 2)
        latency = [120] * 10 + [850, 1400, 2100, 2600, 2300] + [140] * (len(t) - 15)
        ax.plot(t[: len(latency)], latency, color="#C0392B")
        ax.set_title("Payment API p99 Latency (ms) - Incident Window")
        ax.set_xlabel("Minutes since 14:00 UTC")
        ax.set_ylabel("Latency (ms)")
        ax.axvspan(20, 30, color="red", alpha=0.15)
    elif kind == "gantt":
        tasks = ["Discovery", "Design", "Build", "Beta", "GA Launch"]
        starts = [0, 3, 6, 14, 20]
        durations = [3, 4, 9, 5, 2]
        colors = ["#8E44AD", "#2980B9", "#27AE60", "#F39C12", "#C0392B"]
        for i, (s, d, c) in enumerate(zip(starts, durations, colors)):
            ax.barh(tasks[i], d, left=s, color=c)
        ax.set_xlabel("Weeks from kickoff (roadmap starts 2025-01-06)")
        ax.set_title("Product Roadmap 2025 - Workstream Gantt")
    ax.figure.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_network_diagram(out_path):
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=120)
    ax.axis("off")
    nodes = {
        "Edge LB\n(nginx-edge-01)": (0.1, 0.5),
        "API GW\n(gw-cluster-b)": (0.4, 0.7),
        "Auth Svc\n(auth-prod-3)": (0.4, 0.3),
        "Payments DB\n(pg-primary-eu2)": (0.75, 0.7),
        "Cache\n(redis-shard-9)": (0.75, 0.3),
    }
    edges = [
        ("Edge LB\n(nginx-edge-01)", "API GW\n(gw-cluster-b)"),
        ("Edge LB\n(nginx-edge-01)", "Auth Svc\n(auth-prod-3)"),
        ("API GW\n(gw-cluster-b)", "Payments DB\n(pg-primary-eu2)"),
        ("Auth Svc\n(auth-prod-3)", "Cache\n(redis-shard-9)"),
    ]
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax.plot([xa, xb], [ya, yb], color="gray", lw=1.5, zorder=1)
    for name, (x, y) in nodes.items():
        ax.scatter([x], [y], s=1800, color="#3B6EA5", zorder=2)
        ax.text(x, y, name, ha="center", va="center", fontsize=7, color="white", zorder=3)
    ax.set_title("Payments Path - Production Network Diagram")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_expense_table_image(out_path):
    """Simulates a 'scanned table' -- numbers exist only as pixels, not text."""
    fig, ax = plt.subplots(figsize=(5, 2.6), dpi=120)
    ax.axis("off")
    rows = [
        ["Category", "Budgeted", "Actual", "Variance"],
        ["Travel", "$18,000", "$24,350", "+$6,350"],
        ["Software Licenses", "$42,000", "$39,120", "-$2,880"],
        ["Contractor Fees", "$65,000", "$71,900", "+$6,900"],
        ["Office Supplies", "$3,500", "$2,910", "-$590"],
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    ax.set_title("Q3 Expense Report - Finance Dept (scanned)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_floorplan_image(out_path):
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    racks = [("Rack A3", 1, 6), ("Rack A4", 3, 6), ("Rack B1", 1, 3), ("Rack B2", 3, 3),
              ("Rack C7", 6, 4.5), ("Cooling Unit", 8, 6)]
    for label, x, y in racks:
        color = "#E67E22" if "Cooling" in label else "#2C3E50"
        ax.add_patch(plt.Rectangle((x, y), 1.5, 1.2, color=color))
        ax.text(x + 0.75, y + 0.6, label, ha="center", va="center", color="white", fontsize=7)
    ax.text(0.5, 7.5, "Server Room 2 - Floor Plan (fire exit: south wall)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_org_chart_image(out_path):
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=120)
    ax.axis("off")
    boxes = {
        "Vikram Shah\nEngineering Manager": (0.5, 0.85),
        "Rahul Verma\nBackend Lead": (0.2, 0.5),
        "Sneha Pillai\nDevOps Lead": (0.5, 0.5),
        "Karthik Iyer\nML Lead": (0.8, 0.5),
        "Divya Krishnan\nML Engineer": (0.8, 0.15),
    }
    edges = [("Vikram Shah\nEngineering Manager", "Rahul Verma\nBackend Lead"),
             ("Vikram Shah\nEngineering Manager", "Sneha Pillai\nDevOps Lead"),
             ("Vikram Shah\nEngineering Manager", "Karthik Iyer\nML Lead"),
             ("Karthik Iyer\nML Lead", "Divya Krishnan\nML Engineer")]
    for a, b in edges:
        xa, ya = boxes[a]; xb, yb = boxes[b]
        ax.plot([xa, xb], [ya, yb], color="gray", lw=1.2)
    for name, (x, y) in boxes.items():
        ax.scatter([x], [y], s=2200, color="#16A085")
        ax.text(x, y, name, ha="center", va="center", fontsize=6.5, color="white")
    ax.set_title("Engineering Org Chart - AI/Platform")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_pdf(doc_id, filename, title, paragraphs, chart_path=None, chart_caption=""):
    path = os.path.join(PDF_DIR, filename)
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    for para in paragraphs:
        for line in _wrap(para, 95):
            c.drawString(72, y, line)
            y -= 14
        y -= 8
    if chart_path:
        y -= 10
        img = ImageReader(chart_path)
        iw, ih = img.getSize()
        display_w = 380
        display_h = display_w * ih / iw
        if y - display_h < 72:
            c.showPage()
            y = height - 72
        c.drawImage(img, 72, y - display_h, width=display_w, height=display_h)
        y -= display_h + 14
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(72, y, chart_caption)
    c.save()
    return path


def _wrap(text, width):
    import textwrap
    return textwrap.wrap(text, width) or [""]


def main():
    # -- 1. Q3 Sales Report (chart-only fact: which region led revenue, exact $) --
    chart1 = os.path.join(IMG_DIR, "_chart_revenue.png")
    make_chart_png(chart1, "bar_revenue")
    text1 = (
        "This report summarizes company performance for Q3. Overall revenue grew "
        "compared to Q2, driven by strong demand across most regions. The sales "
        "team closed several enterprise deals this quarter, and churn remained "
        "flat. See the chart below for the regional breakdown. Leadership will "
        "discuss regional strategy in the Q4 planning offsite."
    )
    p1 = make_pdf("doc01", "doc01_q3_sales_report.pdf", "Q3 Sales Report",
                  [text1], chart1, "Figure 1: Q3 revenue by region ($M).")
    add_manifest("doc01", "pdf", p1, text1,
                 visual_only_facts=["APAC had the highest Q3 revenue at $6.4M",
                                     "East region was second at $5.8M"],
                 image_paths=[chart1])

    # -- 2. Engineering Onboarding Guide (pure text, no visual) --
    text2 = (
        "Welcome to the Engineering team. On your first day, IT will provision "
        "your laptop and accounts. Please complete the security training module "
        "within your first week. Your manager will assign a buddy for your first "
        "30 days. All new engineers must read the Incident Response runbook and "
        "the Coding Standards guide before shipping their first change. Slack "
        "channels to join: #eng-general, #eng-oncall, #eng-random."
    )
    p2 = make_pdf("doc02", "doc02_onboarding_guide.pdf", "Engineering Onboarding Guide", [text2])
    add_manifest("doc02", "pdf", p2, text2, visual_only_facts=[])

    # -- 3. Network diagram (visual-only: which service sits between GW and cache) --
    chart3 = os.path.join(IMG_DIR, "_diagram_network.png")
    make_network_diagram(chart3)
    text3 = (
        "This document describes the production network path for payment "
        "traffic. Traffic enters through the edge load balancer and is routed to "
        "backend services. See the architecture diagram for the full topology. "
        "Any change to this path requires sign-off from the Platform team."
    )
    p3 = make_pdf("doc03", "doc03_network_architecture.pdf", "Payments Network Architecture",
                  [text3], chart3, "Figure 1: Production network diagram, payments path.")
    add_manifest("doc03", "pdf", p3, text3,
                 visual_only_facts=["Auth service (auth-prod-3) connects directly to the redis cache (redis-shard-9)",
                                     "The API gateway host is gw-cluster-b",
                                     "The payments DB host is pg-primary-eu2"],
                 image_paths=[chart3])

    # -- 4. Employee Benefits FAQ (pure text) --
    text4 = (
        "Q: When does health insurance start? A: Coverage begins on your first "
        "day of employment. Q: How much PTO do I get? A: Full-time employees "
        "accrue 18 days per year, prorated in your first year. Q: Is there a "
        "401k match? A: Yes, the company matches 4% of your contributions. "
        "Contact hr@corp.com with further questions."
    )
    p4 = make_pdf("doc04", "doc04_benefits_faq.pdf", "Employee Benefits FAQ", [text4])
    add_manifest("doc04", "pdf", p4, text4, visual_only_facts=[])

    # -- 5. Server room floor plan (visual only: rack labels, fire exit) --
    floorplan = os.path.join(IMG_DIR, "doc05_floorplan.png")
    make_floorplan_image(floorplan)
    add_manifest("doc05", "image", floorplan,
                 text="", visual_only_facts=[
                     "Rack C7 is the physically isolated rack near the cooling unit",
                     "The fire exit for Server Room 2 is on the south wall",
                 ], image_paths=[floorplan])

    # -- 6. Incident Postmortem (visual only: exact spike timing/duration) --
    chart6 = os.path.join(IMG_DIR, "_chart_latency.png")
    make_chart_png(chart6, "latency_line")
    text6 = (
        "On the afternoon of the incident, the Payments API experienced elevated "
        "latency. Customers reported slow checkout. The on-call engineer was "
        "paged and mitigated the issue by restarting the affected pods. Root "
        "cause was a connection pool exhaustion triggered by a deploy. See the "
        "attached latency chart for the incident timeline. A follow-up action "
        "item was created to add pool-size alerting."
    )
    p6 = make_pdf("doc06", "doc06_incident_postmortem.pdf", "Incident Postmortem: Payments Outage",
                  [text6], chart6, "Figure 1: p99 latency during the incident window.")
    add_manifest("doc06", "pdf", p6, text6,
                 visual_only_facts=["The latency spike lasted roughly 10 minutes, from minute 20 to minute 30",
                                     "Peak p99 latency reached approximately 2600ms"],
                 image_paths=[chart6])

    # -- 7. Expense report (scanned-table style, all numbers visual-only) --
    exp_img = os.path.join(IMG_DIR, "_table_expenses.png")
    make_expense_table_image(exp_img)
    text7 = "Attached is the Q3 expense summary for the Finance department, prepared for the monthly close review."
    p7 = make_pdf("doc07", "doc07_expense_report.pdf", "Q3 Expense Report - Finance",
                  [text7], exp_img, "Table 1: Budget vs. actual by category (scanned).")
    add_manifest("doc07", "pdf", p7, text7,
                 visual_only_facts=["Contractor Fees had the largest overage, +$6,900 over budget",
                                     "Actual travel spend was $24,350 against an $18,000 budget"],
                 image_paths=[exp_img])

    # -- 8. Standalone image: org chart --
    org_img = os.path.join(IMG_DIR, "doc08_org_chart.png")
    make_org_chart_image(org_img)
    add_manifest("doc08", "image", org_img, text="",
                 visual_only_facts=["Divya Krishnan reports to Karthik Iyer, the ML Lead",
                                     "Vikram Shah is the Engineering Manager over Backend, DevOps, and ML leads"],
                 image_paths=[org_img])

    # -- 9. IT Security Policy (pure text) --
    text9 = (
        "All employees must use multi-factor authentication for corporate "
        "accounts. Laptops must have full-disk encryption enabled. Do not store "
        "customer data on personal devices. Report suspected phishing to "
        "security@corp.com immediately. Password managers are provided to all "
        "staff and are mandatory for shared credentials."
    )
    p9 = make_pdf("doc09", "doc09_security_policy.pdf", "IT Security Policy", [text9])
    add_manifest("doc09", "pdf", p9, text9, visual_only_facts=[])

    # -- 10. Product Roadmap (visual only: exact GA launch date/duration) --
    chart10 = os.path.join(IMG_DIR, "_chart_roadmap.png")
    make_chart_png(chart10, "gantt")
    text10 = (
        "This roadmap covers the major workstreams planned for 2025, from early "
        "discovery through general availability. Cross-functional teams will "
        "sync weekly. See the Gantt chart for the full sequencing and "
        "durations of each phase."
    )
    p10 = make_pdf("doc10", "doc10_product_roadmap.pdf", "Product Roadmap 2025",
                   [text10], chart10, "Figure 1: Roadmap workstream Gantt chart.")
    add_manifest("doc10", "pdf", p10, text10,
                 visual_only_facts=["The Build phase is the longest workstream at 9 weeks",
                                     "GA Launch begins in week 20 of the roadmap, which starts 2025-01-06"],
                 image_paths=[chart10])

    # -- 11. Remote Work Policy (pure text) --
    text11 = (
        "Employees may work remotely up to 3 days per week with manager "
        "approval. Core collaboration hours are 10am-3pm in your local time "
        "zone. Equipment stipends of $500/year are available for home office "
        "setup. International remote work requires a separate compliance "
        "review before approval."
    )
    p11 = make_pdf("doc11", "doc11_remote_work_policy.pdf", "Remote Work Policy", [text11])
    add_manifest("doc11", "pdf", p11, text11, visual_only_facts=[])

    # -- 12. Data Retention Policy (pure text) --
    text12 = (
        "Customer transaction records are retained for 7 years to satisfy "
        "regulatory requirements. Application logs are retained for 90 days. "
        "Backups are encrypted at rest and rotated monthly. Deletion requests "
        "under privacy regulations must be fulfilled within 30 days."
    )
    p12 = make_pdf("doc12", "doc12_data_retention_policy.pdf", "Data Retention Policy", [text12])
    add_manifest("doc12", "pdf", p12, text12, visual_only_facts=[])

    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest)} documents. Manifest -> data/manifest.json")


if __name__ == "__main__":
    main()
