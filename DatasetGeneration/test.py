import os

os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

document = "AIMS The aims of this analysis were to examine levels of unmet needs and depression among carers of people newly diagnosed with cancer and to identify groups who may be at higher risk, by examining relationships with demographic characteristics. METHODS One hundred and fifty dyads of people newly diagnosed with cancer and their carers, aged 18 years and older, were recruited from four Australian hospitals. People with cancer receiving adjuvant cancer treatment with curative intent, were eligible to participate. Carers completed the Supportive Care Needs Survey-Partners & Caregivers (SCNS-P&C45), and both carers and patients completed the Centre of Epidemiologic-Depression Scale (CES-D). RESULTS Overall, 57% of carers reported at least one, 37% at least three, 31% at least five, and 15% at least 10 unmet needs; the most commonly endorsed unmet needs were in the domains of information and health care service needs. Thirty percent of carers and 36% of patients were at risk of clinical depression. A weak to moderate positive relationship was observed between unmet needs and carer depression (r=0.30, p<0.001). Carer levels of unmet needs were significantly associated with carer age, hospital type, treatment type, cancer type, living situation, relationship status (in both uni- and multi-factor analysis); person with cancer age and carer level of education (in unifactor analysis only); but not with carer gender or patient gender (in both uni- and multi-factor analyses). CONCLUSION Findings highlight the importance of developing tailored programmes to systematically assist carers who are supporting patients through the early stages of cancer treatment."

prompt = """Example 1:
Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.
Relevant Query: Is a little caffeine ok during pregnancy?

Example 2:
Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.
Relevant Query: What fruit is native to Australia?

Example 3:
Document: The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day.
Relevant Query: How large is the Canadian military?

Example 4:
Document: {}
Relevant Query:"""

def preprocess_output(string, start_point):
    string = string[start_point:]
    first_new_line = string.find("\n") 
    return string[:first_new_line].strip()

p = prompt.format(document)
input_ids = tokenizer(p, return_tensors="pt").input_ids

for i in range(100):
    generated_ids = model.generate(input_ids, max_length=None, max_new_tokens=32, do_sample=True, top_p=0.9)
    print(preprocess_output(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0], len(p)))
