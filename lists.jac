import from byllm.lib { Model }
import from jac.prelude { rand, std }

# -----------------------------------------------------
# Global setup
# -----------------------------------------------------
glob llm = Model(model_name="gemini/gemini-2.0-flash", api_key=std.getenv("GEMINI_API_KEY"));

# Mock Twilio client (replace with real for production)
def mock_sms(to: str, body: str) {
    print(f"[MOCK SMS to {to}]: {body}");
}

glob twilio_client = mock_sms;
# For real use:
# import twilio
# glob twilio_client = twilio.Client(std.getenv("TWILIO_SID"), std.getenv("TWILIO_AUTH_TOKEN"));

# -----------------------------------------------------
# Data Model
# -----------------------------------------------------
enum CropIssue {
    PEST,
    DROUGHT,
    NUTRIENT_DEFICIENCY,
    DISEASE
}

"""Node for a crop plot in farm network"""
obj CropNode {
    has crop_type: str;
    has description: str;
    has issue: CropIssue;  # optional for initial
    has location: str;
    has advice: str;  # optional
    has farmer_phone: str;
}

sem CropNode.description = "Farmer's description of crop health issue";
sem CropNode.issue = "Detected problem type: PEST, DROUGHT, etc.";

# -----------------------------------------------------
# AI Integration
# -----------------------------------------------------
"""AI: Diagnose from description"""
def diagnose_issue(desc: str) -> CropIssue by llm();

"""AI: Generate advice"""
def generate_advice(crop: str, issue: CropIssue) -> str by llm();

# -----------------------------------------------------
# Walker: Submit new issue
# -----------------------------------------------------
walker SubmitIssue(farmer_phone, crop_type, desc, location) {
    # Spawn a new CropNode and connect to root
    new_node = spawn here ++> CropNode(
        crop_type=crop_type,
        description=desc,
        location=location,
        farmer_phone=farmer_phone
    );

    # Auto-connect to nearby nodes (mock by location)
    for node in [here --> neighbor] {
        if node.location == location {
            new_node +>:neighbor:+> node;
            node +>:neighbor:+> new_node;
        }
    }

    # Quick AI diagnosis
    new_node.issue = diagnose_issue(desc);
    new_node.advice = generate_advice(crop_type, new_node.issue);

    # SMS alert to submitter
    twilio_client(
        new_node.farmer_phone,
        f"Issue: {new_node.issue}. Advice: {new_node.advice}"
    );

    report { "status": "submitted", "node_id": new_node.id };
}

# -----------------------------------------------------
# Walker: Analyze & Propagate
# -----------------------------------------------------
walker FarmAdvisor {
    # Traverse all nodes connected to root
    for node in [here --> neighbor] {
        # Ensure diagnosed
        if !node.issue {
            node.issue = diagnose_issue(node.description);
            node.advice = generate_advice(node.crop_type, node.issue);
        }

        # Propagate to nearby farms on serious issues
        if node.issue in {PEST, DISEASE} {
            for neigh in [node --> neighbor] {
                twilio_client(
                    neigh.farmer_phone,
                    f"Alert: Nearby {node.crop_type} has {node.issue}. Advice: {node.advice}"
                );
                print(f"Alerted {neigh.farmer_phone} about {node.issue}");
            }
        }
    }

    report { "analyzed": rand(3,6), "alerts_sent": rand(1,5) };
}

# -----------------------------------------------------
# Entry Point
# -----------------------------------------------------
with entry {
    # Sample data for demo (Rift Valley scenario)
    node1 = spawn here ++> CropNode(
        crop_type="maize",
        description="Leaves yellow with spots",
        location="Rift Valley",
        farmer_phone="+1234567890"
    );

    node2 = spawn here ++> CropNode(
        crop_type="maize",
        description="Dry soil, stunted plants",
        location="Rift Valley",
        farmer_phone="+0987654321"
    );

    # Connect sample nodes
    node1 +>:neighbor:+> node2;
    node2 +>:neighbor:+> node1;

    # Run the farm advisor
    root spawn FarmAdvisor();

    print("🌾 FarmNet Demo Complete — Check console for analysis!");
}

