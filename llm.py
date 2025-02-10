import re
from openai import OpenAI
import httpx

API_KEY = 'sk-xxxxxxxx'
BASE_URL = 'https://api.openai.com/v1'

class GPTSelectShapelet:
    def __init__(self, information, model='gpt-3.5-turbo', Api=API_KEY):
        self.model = model
        self.information = information
        self.Api = Api
        self.system_prompt = "You are an expert in time series data analysis, especially skilled in time series classification. Now, you will get a shapelet and it`s information. You should evaluate the quality of the shapelet. You also need to make some reasoning for it`s practical significance with the information of the dataset, such as what pattern does this shapelet represent, and what event may its appearance represent. Here is the form of the data you will get:" \
        "[Shapelet] [Raw Data] [Label] [Channel] [Contribution] [Information]" \
        "Here the items are split by space. Shapelet is a practical significance subsequence; Raw Data is the time series data that produced the shapelet; Channel is the dimension of the multivariate time series data; Label is the label of raw data, also the most possible label for shapelet; Contribution is Information Gain of the shapelet, which means that the shapelet`s Score should be lower if the Contribution is too low; Information is a short introduction of the whole datasets." \
        "You should output your evaluation with the form of:" \
        "[Score] [Reasoning]" \
        "Here Score is a integer between 0 and 100, and the higher the score, the more likely it is to improve the accuracy of the classifier by retaining it. Reasoning is a short description text that the most likely be the significance of the shapelet." \
        "Pay attention that only output a integer, space as split and a text, without any explaination or reasoning process as beginning or ending."
        self.error_num = 0


    def gpt_call(self, aim):
        try:
            client = OpenAI(
                base_url=BASE_URL, 
                api_key=self.Api,
                http_client=httpx.Client(
                    base_url=BASE_URL,
                    follow_redirects=True,
                ),
            )
            completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": aim}
            ]
            )
            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"GPT call get an exception as {e}")
            self.error_num += 1
            if self.error_num >= 5:
                print("Too much exception from gpt")
                exit(0)
            else:
                return "0 None"
    
    def load_data_prompt(self, shapelet, raw_data):
        prompt = f"{str(shapelet[6])} {raw_data} {shapelet[3]} {shapelet[5]} {shapelet[0]} {self.information}"
        result = self.gpt_call(prompt)
        score = result.split()[0]
        information = ' '.join(result.split()[1:])
        match = re.search(r'-?\d+', score)
        if match:
            score = int(match.group())
        else:
            score = 0
        print('=' * 50)
        print(prompt + '\n')
        print(score)
        print(information)
        print('=' * 50)
        return score, information

