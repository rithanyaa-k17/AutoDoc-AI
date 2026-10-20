"""Generate two SYNTHETIC AUTOSAR-style HLD PDFs (v1.0 and v1.1) + a ground-truth JSON.

All names are fictional. These files are test data only: NOT official AUTOSAR documents and
NOT Tata Technologies material.   Run from the project root:  python Code/generate_synthetic.py
"""
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "Input_Data"
ss = getSampleStyleSheet()
H1, H2, BODY = ss["Heading1"], ss["Heading2"], ParagraphStyle("b", parent=ss["BodyText"], fontSize=10.5, leading=15)
NOTE = ParagraphStyle("n", parent=BODY, textColor="#aa0000")


def doc_pages(version):
    v11 = version == "1.1"
    p = []
    p.append([("t", "Synthetic Smart Wiper System - High-Level Design"),
              ("n", "SYNTHETIC TEST DATA - fictional example for tool testing. Not an official AUTOSAR "
                    "document and not Tata Technologies material."),
              ("b", f"Document ID: SYN-HLD-WIPER-001. Version: {version}."),
              ("h1", "1 Introduction"), ("h2", "1.1 Purpose"),
              ("b", "This synthetic High-Level Design describes a fictional Smart Wiper system. "
                    "It exists only to test the AutoDoc AI tool."),
              ("h2", "1.2 Scope"),
              ("b", "The scope covers rain detection, wiper speed selection, motor driving and fault storage.")])
    comp = [("h1", "2 System Overview"), ("h2", "2.1 Software Components"),
            ("b", "WiperControlSwc coordinates the wiper speed request."),
            ("b", "RainSensorSwc measures rain intensity and provides it through RainSensorInterface."),
            ("b", "WiperMotorDriverSwc drives the wiper motor and reports faults."),
            ("b", "VehicleSpeedProviderSwc supplies the vehicle speed."),
            ("b", "DiagnosticHandlerSwc stores wiper faults.")]
    if v11:
        comp.append(("b", "WiperPositionSensorSwc measures the wiper blade position for WiperControlSwc."))
    p.append(comp)
    p.append([("h1", "3 Interfaces and Ports"), ("h2", "3.1 RainSensorInterface"),
              ("b", "The port P_RainIntensity is provided by RainSensorSwc."),
              ("b", "The port R_RainIntensity is required by WiperControlSwc."),
              ("b", "The signal Sig_RainIntensity is updated every 20 ms."),
              ("h2", "3.2 MotorCommandInterface"),
              ("b", "The port P_MotorCommand is provided by WiperControlSwc."),
              ("b", "The port R_MotorCommand is required by WiperMotorDriverSwc."),
              ("b", "The signal Sig_MotorSpeedCmd carries the requested motor speed."),
              ("h2", "3.3 SpeedInterface"),
              ("b", "The signal Sig_VehicleSpeed is provided by VehicleSpeedProviderSwc."),
              ("b", "RainSensorIf is the legacy name used by the diagnostic tooling.")])
    flow = [("h1", "4 Functional Flows"), ("h2", "4.1 Automatic Wiping Flow"),
            ("b", "First, RainSensorSwc sends Sig_RainIntensity to WiperControlSwc every 50 ms."),
            ("b", "Then WiperControlSwc requests the current vehicle speed from VehicleSpeedProviderSwc."),
            ("b", "Next, WiperControlSwc sends Sig_MotorSpeedCmd to WiperMotorDriverSwc."),
            ("h2", "4.2 Fault Handling Flow"),
            ("b", "If WiperMotorDriverSwc detects an overcurrent, it reports a fault to DiagnosticHandlerSwc."),
            ("b", "After a fault is reported, DiagnosticHandlerSwc notifies WasherPumpSwc and stores the fault.")]
    if not v11:
        flow.append(("b", "The WiperMotorDriverSwc performs a self-test at startup."))
    p.append(flow)
    req = [("h1", "5 Requirements"),
           ("b", "HLD-FUN-001: WiperControlSwc shall select the wiper speed from the rain intensity."
                 if not v11 else
                 "HLD-FUN-001: WiperControlSwc shall select the wiper speed from the rain intensity and "
                 "the vehicle speed."),
           ("b", "HLD-FUN-002: WiperMotorDriverSwc shall limit the motor current to 8 A."
                 if not v11 else
                 "HLD-FUN-002: WiperMotorDriverSwc shall limit the motor current to 10 A."),
           ("b", "HLD-FUN-002: DiagnosticHandlerSwc shall store each wiper fault.")]
    if v11:
        req.append(("b", "HLD-FUN-004: WiperPositionSensorSwc shall report the blade position."))
    p.append(req)
    return p


def build(version, path):
    def footer(c, d):
        c.setFont("Helvetica", 8)
        c.drawString(40, 25, f"SYNTHETIC TEST DATA - not an official AUTOSAR document | v{version} | Page {d.page}")

    story = []
    for i, page in enumerate(doc_pages(version)):
        for kind, text in page:
            style = {"t": H1, "h1": H1, "h2": H2, "b": BODY, "n": NOTE}[kind]
            story.append(Paragraph(text, style))
            story.append(Spacer(1, 4))
        if i < 4:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4, title=f"Synthetic Wiper HLD v{version}",
                      author="Synthetic test data generator").build(story, onFirstPage=footer, onLaterPages=footer)


GROUND_TRUTH = {
    "_note": "Authored together with the generator; a self-made benchmark, not an independent one.",
    "components": ["WiperControlSwc", "RainSensorSwc", "WiperMotorDriverSwc", "VehicleSpeedProviderSwc",
                   "DiagnosticHandlerSwc", "WasherPumpSwc"],
    "interfaces": ["RainSensorInterface", "MotorCommandInterface", "SpeedInterface", "RainSensorIf"],
    "ports": ["P_RainIntensity", "R_RainIntensity", "P_MotorCommand", "R_MotorCommand"],
    "signals": ["Sig_RainIntensity", "Sig_MotorSpeedCmd", "Sig_VehicleSpeed"],
    "qa": [  # question, expected page (v1.0), phrase that must appear in retrieved evidence
        {"q": "Which component measures rain intensity?", "pages": [2], "phrase": "RainSensorSwc measures rain intensity"},
        {"q": "Which port is provided by RainSensorSwc?", "pages": [3], "phrase": "P_RainIntensity"},
        {"q": "What happens when WiperMotorDriverSwc detects an overcurrent?", "pages": [4], "phrase": "overcurrent"},
        {"q": "What is the motor current limit?", "pages": [5], "phrase": "limit the motor current"},
        {"q": "How does WiperControlSwc request vehicle speed?", "pages": [4], "phrase": "requests the current vehicle speed"},
        {"q": "Which signal carries the requested motor speed?", "pages": [3], "phrase": "Sig_MotorSpeedCmd"},
        {"q": "What is the purpose of this document?", "pages": [1], "phrase": "Smart Wiper system"},
    ],
    "negative_questions": ["What is the recipe for chocolate cake?", "Explain quantum chromodynamics lagrangian.",
                           "Who won the football world cup?"],
    "expected_changes_v1_0_to_v1_1": [
        {"type": "modified", "contains": "HLD-FUN-001"},
        {"type": "modified", "contains": "motor current"},
        {"type": "added", "contains": "WiperPositionSensorSwc measures"},
        {"type": "added", "contains": "HLD-FUN-004"},
        {"type": "removed", "contains": "self-test"},
    ],
    "expected_affected_elements": ["WiperControlSwc", "WiperMotorDriverSwc", "WiperPositionSensorSwc"],
    "expected_inconsistency_rules_v1_0": {
        "R1": "HLD-FUN-002", "R2": "WasherPumpSwc", "R3": "RainSensorIf", "R4": "Sig_RainIntensity"},
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for v in ("1.0", "1.1"):
        f = OUT / f"SYNTHETIC_Wiper_HLD_v{v}.pdf"
        build(v, f)
        print("created", f)
    (OUT / "synthetic_ground_truth.json").write_text(json.dumps(GROUND_TRUTH, indent=2), encoding="utf-8")
    print("created", OUT / "synthetic_ground_truth.json")
