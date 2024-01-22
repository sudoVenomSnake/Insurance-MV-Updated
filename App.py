import json
import os
import redirect as rd

from llama_index import StorageContext, load_index_from_storage, ServiceContext
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.question_gen.guidance_generator import GuidanceQuestionGenerator
from llama_index.prompts.guidance_utils import convert_to_handlebars
from llama_index.question_gen.types import SubQuestion
from guidance.models import OpenAI as GuidanceOpenAI

import streamlit as st

st.set_page_config(layout = "wide")

st.session_state.key = st.secrets["OPENAI_API_KEY"]

if "key" not in st.session_state:
    os.environ["OPENAI_API_KEY"] = st.text_input(label = "Please enter your OpenAI key")
    if os.environ["OPENAI_API_KEY"]:
        st.session_state.key = os.environ["OPENAI_API_KEY"]

if "loaded" not in st.session_state:
    with open("template.json", "r") as f:
        template = json.load(f)
    st.session_state.title = template["title"]
    st.session_state.index_files = template["index_names"]
    st.session_state.summaries = template["summaries"]
    st.session_state.model_choice = template["model_choice"]
    st.session_state.loaded = True

def initialize():
    query_engine_tools = []

    def build_tools_text(tools) -> str:
        tools_dict = {}
        for tool in tools:
            tools_dict[tool.name] = tool.description
        return json.dumps(tools_dict, indent=4)


    PREFIX = """\
    Given a user question, and a list of tools, output a list of relevant sub-questions (don't ask generic questions about the document until the user question needs it) \
    in json markdown that when composed can help answer the full user question: 

    """


    example_query_str = (
        "An employee at a manufacturing company has been diagnosed with a chronic health illness which would require consistent medical treatment and eventual organ transplantation."
    )
    example_tools = [
        ToolMetadata(
            name="THE EMPLOYEES\u2019 STATE INSURANCE ACT, 1948",
            description="The Employees' State Insurance Act, 1948, is a comprehensive social security legislation in India that provides a range of benefits to employees in cases of sickness, maternity, and employment injury. The Act applies to all factories and can be extended to other establishments by the government. It defines key terms such as \"employee,\" \"wages,\" and \"employment injury\" to ensure clarity in its application.\n\nThe Act establishes the Employees' State Insurance Corporation (ESIC), which is responsible for administering the scheme. The ESIC is a corporate body with representatives from the government, employers, employees, and medical professionals. Members serve four-year terms and are responsible for the operation and management of the ESIC, including the Employees' State Insurance Fund, which is financed by contributions from both employers and employees.\n\nEmployers are required to register with the ESIC and insure their employees. Contributions are determined by the Central Government and are used to provide various benefits, including medical treatment and cash benefits for sickness, maternity, and employment injuries. The Act also outlines the responsibilities of employers in maintaining records and facilitating compliance.\n\nThe ESIC has the authority to establish medical facilities and educational institutions to improve service quality. Benefits are non-transferable and protected from attachment, and the Act prohibits the receipt of similar benefits from other sources. The Act also includes provisions for the recovery of benefits paid in error and for the adjudication of disputes by the Employees' Insurance Court.\n\nPenalties for non-compliance with the Act's provisions include fines and imprisonment. The Act allows for the recovery of damages from employers who fail to pay contributions and provides for the adjudication of disputes and claims related to employee insurance in the Employees' Insurance Court, with appeals to the High Court on substantial questions of law.",
        ),
        ToolMetadata(
            name="Insurance Act,1938 - incorporating all amendments",
            description="The Insurance Act of 1938, incorporating all amendments up to 2021, is a comprehensive document outlining the regulations and provisions for insurers in India. The Act covers a wide range of topics including the appointment of a Controller of Insurance, registration requirements, capital structure requirements, and the licensing of agents. It also details the process for amalgamation and transfer of insurance business, and the management of insurance business by an administrator. The Act stipulates that an insurance company must have a minimum paid-up capital of one hundred crore rupees and that no foreign body corporate can hold more than twenty-six percent of the capital of such a company. The Act also outlines the conditions under which an insurer's registration may be suspended or cancelled, and the requirements for insurers and insurance businesses in India. The Act has undergone several amendments, with key changes made in 2015.",
        ),
        ToolMetadata(
            name="INSURANCE REGULATORY AND DEVELOPMENT AUTHORITY OF INDIA (HEALTH INSURANCE) REGULATIONS, 2016",
            description="The document is a PDF of the Insurance Regulatory and Development Authority of India (IRDAI) Health Insurance Regulations, 2016, with amendments up to November 19, 2019. It outlines the regulations established by IRDAI under the powers conferred by the Insurance Act, 1938, and the IRDA Act, 1999, in consultation with the Insurance Advisory Committee.\n\nKey points from the document include:\n\n1. The regulations are titled \"Insurance Regulatory and Development Authority of India (Health Insurance) Regulations, 2016\" and came into effect upon their publication in the official Gazette of India. They apply to all registered life, general, and health insurers conducting health insurance business, as well as Third-Party Administrators (TPAs).\n\n2. Definitions are provided for terms such as \"Act,\" \"Authority,\" \"AYUSH Treatment,\" \"Break in policy,\" \"Cashless facility,\" \"Health Services Agreement,\" \"Health insurance business,\" \"Health Services by TPA,\" \"Health plus Life Combi Products,\" \"Network Provider,\" \"Pilot product,\" \"Senior citizen,\" and \"Specified.\"\n\n3. Health insurance products can only be offered by entities with valid registration, and there are specific provisions for life insurers, general insurers, and health insurers regarding the types of health insurance products they can offer and their tenure.\n\n4. The document details the product filing procedure for health insurance products, stating that no product can be marketed or offered without being filed with the Authority as per the Product Filing Guidelines.\n\n5. The withdrawal of health insurance products is subject to guidelines specified by the Authority, and life insurers must withdraw indemnity-based products within three months of the notification of these regulations.\n\n6. Health insurance products must be reviewed at least annually by the Appointed Actuary, and if found financially unviable or deficient, they may be revised.\n\n7. Group health insurance policies must not be issued solely for the purpose of insurance, and there must be a clear relationship between group members and the policyholder. The minimum group size is determined by the insurer, subject to a minimum of 7 members.\n\n8. Insurers must have a Health Insurance Underwriting Policy approved by their Board, which should be reviewed periodically. The policy should cover standard and sub-standard lives and set out the underwriting parameters. Proposals can be accepted, modified, or denied based on this policy, and denials must be communicated in writing.\n\nThe document also mentions that general insurers and health insurers may offer incentives to policyholders for early entry, renewals, favorable claims experience, and preventive and wellness habits, which must be disclosed in the prospectus and policy document.",
        ),
        ToolMetadata(
            name="THE TRANSPLANTATION OF HUMAN ORGANS AND TISSUES ACT, 1994",
            description="The Transplantation of Human Organs and Tissues Act, 1994, is an Indian legislation enacted to regulate the removal, storage, and transplantation of human organs and tissues for therapeutic purposes and to prevent commercial dealings in them. The Act, numbered 42 of 1994, was passed on July 8, 1994, and initially applied to the states of Goa, Himachal Pradesh, Maharashtra, and all Union territories. Other states could adopt the Act via resolution under article 252 of the Constitution.\n\nThe Act is structured into several chapters, detailing preliminary information, authority for organ and tissue removal, regulation of hospitals, the appropriate authority and advisory committees, registration of hospitals and tissue banks, offences and penalties, and miscellaneous provisions.\n\nKey definitions include \"donor\" as a person authorizing organ or tissue removal, \"hospital\" encompassing various medical institutions, \"human organ\" as a part of the body with structured tissues, \"near relative\" including family members, \"recipient\" as the person receiving the organ or tissue, and \"transplantation\" as the grafting of organs or tissues for therapeutic purposes.\n\nThe Act outlines the conditions under which human organs or tissues can be removed, including the consent of the donor, and establishes an appropriate authority for overseeing the process. It also sets regulations for hospitals conducting removal, storage, or transplantation, and mandates the registration of such hospitals and tissue banks.\n\nOffences and penalties are specified for unauthorized removal, commercial dealings, and contraventions of the Act. The Act provides protection for actions taken in good faith and grants the power to make rules for its implementation. It came into force on February 4, 1995, and was extended to Jammu and Kashmir and Ladakh in 2019.",
        ),
    ]
    example_tools_str = build_tools_text(example_tools)
    example_output = [
        SubQuestion(sub_question="What are the health benefits provided under the Employees' State Insurance Act, 1948 for chronic illnesses?", tool_name="THE EMPLOYEES\u2019 STATE INSURANCE ACT, 1948"),
        SubQuestion(sub_question="Who is eligible for medical advantages under the Employees' State Insurance Act and what is the process for registration?", tool_name="THE EMPLOYEES\u2019 STATE INSURANCE ACT, 1948"),
        SubQuestion(sub_question="What is the role of employers in the Employees' State Insurance scheme?", tool_name="THE EMPLOYEES\u2019 STATE INSURANCE ACT, 1948"),
        SubQuestion(sub_question="What are the legal obligations of insurance companies to cover chronic health issues or terminal illnesses under the Insurance Act of 1938?", tool_name="Insurance Act,1938 - incorporating all amendments"),
        SubQuestion(sub_question="Is there a specific regulation concerning coverage of organ transplants under a health insurance policy in IRDA regulations?", tool_name="INSURANCE REGULATORY AND DEVELOPMENT AUTHORITY OF INDIA (HEALTH INSURANCE) REGULATIONS, 2016"),
        SubQuestion(sub_question="What are the provisions and conditions under which an organ can be removed for a transplant according to the Transplantation of Human Organs and Tissues Act, 1994?", tool_name="THE TRANSPLANTATION OF HUMAN ORGANS AND TISSUES ACT, 1994"),
    ]
    example_output_str = json.dumps({"items": [x.dict() for x in example_output]}, indent=4)

    EXAMPLES = f"""\
    # Example 1
    <Tools>
    ```json
    {example_tools_str}
    ```

    <User Question>
    {example_query_str}


    <Output>
    ```json
    {example_output_str}
    ```

    """.replace(
        "{", "{{"
    ).replace(
        "}", "}}"
    )

    SUFFIX = """\
    # Example 2
    <Tools>
    ```json
    {tools_str}
    ```

    <User Question>
    {query_str}

    <Output>
    """

    subquestion_template = convert_to_handlebars(PREFIX + EXAMPLES + SUFFIX)
    question_gen = GuidanceQuestionGenerator.from_defaults(prompt_template_str = subquestion_template, guidance_llm = GuidanceOpenAI("gpt-4"), verbose = False)

    context = """You are a seasoned insurance attorney AI agent, specializing in the complex landscape of healthcare insurance in the Indian context. You have acess to a tool, which you will use to formulate answers, when given queries. Use the context given in these answers to formulate your final answer, make sure you cite well with statutes et cetera."""
    for i, j in zip(st.session_state.index_files, st.session_state.summaries):
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir = "RAG Generator/" + i), service_context = ServiceContext.from_defaults(llm = OpenAI(temperature = 0, model = st.session_state.model_choice)))
        query_engine_tools.append(QueryEngineTool(
            query_engine = index.as_query_engine(similarity_top_k = 3),
            metadata = ToolMetadata(
                name = i,
                description = (j + " ",
                    "Properly frame question as it will match relevance to compute answer."
                ),
            ),
        ))
    llm = OpenAI(model = st.session_state.model_choice)
    query_engine = SubQuestionQueryEngine.from_defaults(question_gen = question_gen, query_engine_tools = query_engine_tools, verbose = True)
    final_tool = [QueryEngineTool(
            query_engine = query_engine,
            metadata = ToolMetadata(
                name = "Index",
                description = ("This tool is the Master Index and the source for your answers. ",
                    "Add as much information about the query as possible so the tool excels."
                ),
            ),
        )]
    st.session_state.agent = ReActAgent.from_tools(final_tool, context = context, llm = llm, verbose = True)
    return

if "agent" not in st.session_state:
    initialize()

if "agent" in st.session_state:
    st.title(st.session_state.title)
    for i, j in zip(st.session_state.index_files, st.session_state.summaries):
        st.subheader(i)
        st.write(j)
    query = st.text_input(label = "Please enter your query -")
    if query:
        with rd.stdout as out:
            st.session_state.agent.query(query)