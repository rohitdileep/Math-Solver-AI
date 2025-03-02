import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import  AgentType
from langchain.agents import Tool , initialize_agent
from langchain.callbacks import StreamlitCallbackHandler




st.set_page_config(page_title = "Text to math problem and DataSearch assistant")
st.title('Math problem solver using deepseek-r1-distill-qwen-32b')


groq_api_key = st.sidebar.text_input("Groq api key " , type = 'password')

if not groq_api_key :
    st.info("Please input the api key")
    st.stop()

## llm initializing ##
llm  = ChatGroq(model = 'deepseek-r1-distill-qwen-32b' , groq_api_key = groq_api_key)

## initializing the tool ###

wikipedia_rapper =  WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = 'Wikipedia' ,
    func = wikipedia_rapper.run ,
    description = 'Tool used to search wikipedia' 
    )


### initialise the math tool ###

math_chain =  LLMMathChain.from_llm(llm = llm)

calculator_tool = Tool(
    name = 'Calculations' ,
    func = math_chain.run ,
    description ='Tools used to perform mathematical calculations' 
)


prompt = """ You are a agent who task is to do mathematical question.Logically arrive at the solution and 
        display it point wise for the question below.
        Question : {question}
        Answer : """

prompt_template = PromptTemplate(template  = prompt , input_variable = ['question'] )

chain = prompt_template | llm

## Reasoning tool ##
reasoning_tool = Tool(
    name = 'Reasoning' , 
    func = chain.invoke ,
    description  = 'Tool which will be used to expalin concepts'
)

### initializing the agent ###


assistant_agent = initialize_agent(tools = [wikipedia_tool , calculator_tool , reasoning_tool ] ,
                                    llm = llm ,
                                    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION ,
                                    handling_parsing_errors = True ,
                                    verbose = False
                                    )



if 'messages' not in st.session_state :
    st.session_state['messages'] =  [{"role" : "Assistant" , "content" : "Hi i am  a  chatbot i will be assistig you for solving Math related problem"}]

# Initialize reset flag
if 'should_reset' not in st.session_state:
    st.session_state.should_reset = False

# Reset user_input if needed before rendering the widget
if st.session_state.should_reset:
    st.session_state.user_input = ""
    st.session_state.should_reset = False

for msg in st.session_state['messages']:
    st.chat_message(msg["role"]).write(msg["content"])



###  interaction ###
# Input Widget
question = st.text_area(
    "Enter your question?",
    key="user_input",
    value=st.session_state.get("user_input", "")
)

def response():
    st.session_state.messages.append({"role" : "user" , "content" : question })
    st.chat_message("user").write(question)
    st_cb = StreamlitCallbackHandler(st.container() , expand_new_thoughts = False)
    response = assistant_agent.run(st.session_state.messages , callbacks = [st_cb])
    st.session_state.messages.append({"role" : "Assistant" , "content" : response  })
    return response

if st.button('Find my answer') :
    if question :
        with st.spinner('Generate Response') :

            response =  response()
            st.write("### Response")
            st.success(response)
            st.session_state.should_reset = True 
        

        # **Trigger re-render for the new question input field**
           
        st.rerun()

    else :
        st.warning('Please enter a question')

