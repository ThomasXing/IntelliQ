# encoding=utf-8
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from scene_config import scene_prompts
from utils.date_utils import get_current_date

# 创建意图关联性判断的提示模板
intent_relation_template = PromptTemplate(
    input_variables=["scene_description", "user_input"],
    template="判断当前用户输入内容与当前对话场景的关联性:\n\n当前对话场景: {scene_description}\n当前用户输入: {user_input}\n\n这两次输入是否关联（仅用小数回答关联度，得分范围0.0至1.0）"
)

# 创建意图识别的提示模板
intent_recognition_template = PromptTemplate(
    input_variables=["options_prompt", "user_input"],
    template="有下面多种场景，需要你根据用户输入进行判断，只答选项\n{options_prompt}\n用户输入：{user_input}\n请回复序号："
)

# 创建槽位更新的提示模板
slot_update_template = PromptTemplate.from_template(
    scene_prompts.slot_update
)

# 创建槽位查询的提示模板
slot_query_template = PromptTemplate(
    input_variables=["scene_name", "slot_json", "user_input"],
    template=scene_prompts.slot_query_user
)

# 工厂函数，用于创建槽位更新消息
def create_slot_update_prompt(scene_name, dynamic_example, slot_template, user_input):
    return slot_update_template.format(
        scene_name=scene_name,
        current_date=get_current_date(),
        dynamic_example=dynamic_example,
        slot_json=slot_template,
        user_input=user_input
    )

# 工厂函数，用于创建槽位查询消息
def create_slot_query_prompt(scene_name, slot, user_input):
    return slot_query_template.format(
        scene_name=scene_name,
        slot_json=slot,
        user_input=user_input
    )