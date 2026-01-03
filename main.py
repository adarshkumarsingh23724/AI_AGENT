import os, io, re, json
from datetime import datetime
from flask import Flask, render_template, request, send_file
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import tools

load_dotenv()
app = Flask(__name__)

class DeepResearchResponse(BaseModel):
    topic: str
    summary: str = Field(description="A deep technical summary.")
    findings: list[str] = Field(description="Detailed bullet points.")
    demographics: list[str] = Field(description="Statistical or demographic data.")
    sources: list[str] = Field(description="List of URLs or sources used.")

parser = PydanticOutputParser(pydantic_object=DeepResearchResponse)

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0, 
    model_kwargs={"tool_choice": "auto"}
)

def format_to_text(data):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = [f"--- Research Output ---\nTimestamp: {ts}\n", f"{data.topic} Overview\n", f"{data.summary}\n", "\nKey Findings:"]
    report.extend([f" - {f}" for f in data.findings])
    report.append("\nDemographics/Data:")
    report.extend([f" - {d}" for d in data.demographics])
    report.append("\nSources:")
    report.extend([f" - {s}" for s in data.sources])
    return "\n".join(report)

@app.route("/", methods=["GET", "POST"])
def index():
    report = None
    if request.method == "POST":
        query = request.form.get("query")
        system_prompt = f"""You are an Elite Researcher. 
        1. Use tools to gather data. 
        2. NEVER wrap tool calls in XML tags like <function>. 
        3. Your final response must be ONLY a JSON object:
        {parser.get_format_instructions()}"""
        
        agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
        
        try:
            result = agent.invoke({"messages": [("human", query)]})
            content = result["messages"][-1].content
            
            # Extract only the last JSON block (ignores tool calling logs)
            json_blocks = re.findall(r'(\{.*?\})', content, re.DOTALL)
            structured_data = None
            for block in reversed(json_blocks):
                try:
                    structured_data = parser.parse(block)
                    break
                except: continue
                
            report = format_to_text(structured_data) if structured_data else f"Parse Error. Content: {content}"
        except Exception as e:
            report = f"--- Error ---\n{str(e)}"
            
    return render_template("index.html", report=report)

@app.route("/download", methods=["POST"])
def download():
    content = request.form.get("content")
    buf = io.BytesIO(content.encode("utf-8"))
    return send_file(buf, as_attachment=True, download_name="research.txt", mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)