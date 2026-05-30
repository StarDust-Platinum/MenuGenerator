import os
import json
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel

# 页面基本配置
st.set_page_config(page_title="食材库存管理与智能菜单推荐", layout="wide")

INV_PATH = "data/inventory.csv"
CAT_PATH = "data/category.csv"

# 确保必要的数据目录存在
os.makedirs("data", exist_ok=True)

# 加载静态数据规格
@st.cache_data
def load_categories():
    if os.path.exists(CAT_PATH):
        return pd.read_csv(CAT_PATH)
    else:
        st.error(f"基础数据文件不存在: {CAT_PATH}，请检查项目文件结构。")
        return pd.DataFrame(columns=["食材", "单位质量/g"])

df_cat = load_categories()
VALID_INGREDIENTS = df_cat["食材"].tolist()
CAT_MAP = dict(zip(df_cat["食材"], df_cat["单位质量/g"]))

# 初始化与加载库存数据
if not os.path.exists(INV_PATH):
    # 数据文件不存在时，创建文件和表头但不添加记录
    df_empty = pd.DataFrame(columns=["食材", "单位质量", "份数"])
    df_empty.to_csv(INV_PATH, index=False)

def load_inventory():
    df = pd.read_csv(INV_PATH)
    # 表格显示时按数据规格定义中的顺序排序
    df['sort_idx'] = df['食材'].apply(lambda x: VALID_INGREDIENTS.index(x) if x in VALID_INGREDIENTS else 999)
    df = df.sort_values('sort_idx').drop(columns=['sort_idx']).reset_index(drop=True)
    return df

def save_inventory(df):
    df.to_csv(INV_PATH, index=False)

df_inv = load_inventory()

# 初始化 Gemini API 客户端
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# ==============================================================================
# UI 区域一：食材库存数据与数据操作区域
# ==============================================================================
st.title("🍳 食材库存管理与菜单推荐系统")
st.header("📦 食材库存管理")

# 列表渲染、编辑与删除
if not df_inv.empty:
    for idx, row in df_inv.iterrows():
        # 增加 vertical_alignment="bottom"，使文本、输入框和删除按钮横向完美对齐
        cols = st.columns([3, 2, 2, 2], vertical_alignment="bottom")
        
        cols[0].markdown(f"### {row['食材']}") 
        cols[1].markdown(f"**{row['单位质量']} g / 份**")
        
        # 编辑数量逻辑 (最低消耗或起步为 1 份，更改后立即写入文件)
        new_qty = cols[2].number_input(
            "份数", 
            min_value=1, 
            value=int(row['份数']), 
            key=f"qty_{idx}"
        )
        if new_qty != row['份数']:
            df_inv.at[idx, '份数'] = new_qty
            save_inventory(df_inv)
            st.rerun()
            
        # 删除记录逻辑：更改后立即写入文件
        if cols[3].button("🗑️ 删除", key=f"del_{idx}", use_container_width=True):
            df_inv = df_inv.drop(idx).reset_index(drop=True)
            save_inventory(df_inv)
            st.rerun()
else:
    st.info("当前库存中没有记录，请在下方表单添加。")

st.markdown("---")
st.subheader("➕ 添加新食材记录")

# 过滤掉当前已经存在于库存中的食材，避免重复添加，新添加的记录必须在规格定义内
available_to_add = [x for x in VALID_INGREDIENTS if x not in df_inv["食材"].values]

# 使用 st.form，写入后自动清空表单
with st.form("add_ingredient_form", clear_on_submit=True):
    # 增加 vertical_alignment="bottom"，使表单输入框和提交按钮底部对齐
    f_cols = st.columns([4, 2, 2], vertical_alignment="bottom")
    
    selected_ing = f_cols[0].selectbox("选择食材", options=available_to_add, index=0 if available_to_add else None)
    qty_to_add = f_cols[1].number_input("份数", min_value=1, value=1, step=1)
    
    # 使用标准标准的 form_submit_button 触发表单提交
    submit_btn = f_cols[2].form_submit_button("添加", use_container_width=True)
    
    if submit_btn and selected_ing:
        unit_w = CAT_MAP[selected_ing]
        new_row = pd.DataFrame([{"食材": selected_ing, "单位质量": unit_w, "份数": qty_to_add}])
        df_inv = pd.concat([df_inv, new_row], ignore_index=True)
        save_inventory(df_inv)
        st.rerun()


# ==============================================================================
# UI 区域二：菜单推荐与菜谱显示区域
# ==============================================================================
st.markdown("---")
st.header("💡 智能菜单推荐与菜谱")

if not client:
    st.warning("环境变量 `GEMINI_API_KEY` 未配置，请在 docker-compose 中设置后使用推荐功能。")
else:
    # 获取有效库存
    current_stock = {row['食材']: int(row['份数']) for idx, row in df_inv.iterrows() if row['份数'] > 0}

    # 核心修改：进入或刷新页面后，不直接查询菜单，只显示生成菜单的按钮
    if st.button("🔄 生成菜单推荐", use_container_width=True):
        if not current_stock:
            st.session_state['recommendations'] = []
            st.warning("当前没有可用的库存食材，请先添加食材。")
        else:
            with st.spinner("Gemini 2.5-flash 正在根据库存定制营养膳食..."):
                # 构建严格的 Prompt 业务契约
                prompt = f"""
                你是一个专业的家庭营养配餐师。请基于以下提供的库存食材，推荐 3 组完全不同的菜单方案。
                
                当前可用食材库存 (食材名称: 剩余份数):
                {json.dumps(current_stock, ensure_ascii=False)}
                
                食材与其对应的单位质量映射关系:
                {json.dumps(CAT_MAP, ensure_ascii=False)}

                业务逻辑与硬性约束：
                1. 一次必须精准提供【3】组不同的菜单推荐。
                2. 菜单所用食材量必须在当前库存量以内，绝对不能推荐库存不足以制作的菜单。
                3. 食材扣除的最小单位是 1 份，且消耗量必须为整数份。
                4. 目标用餐人群：【2名成人和1名18个月儿童】。要求营养均衡、口味清淡，适合幼儿消化且安全。
                5. 烹饪时间约束：包括前期食材切分、处理在内的总烹饪时间必须在【20分钟以内】。
                6. 菜谱处理逻辑：形态为原状的大体积食材（如整块肉类、排骨、大块根茎蔬菜等）必须在菜谱步骤中明确包含切碎、切小、切丝或去骨等处理说明，以适应18个月幼儿。
                7. 菜谱步骤必须分步骤逐行结构化返回。
                """

                # 使用 Pydantic 的 BaseModel 定义结构化输出 Schema
                class RecipeStep(BaseModel):
                    step_num: int
                    description: str

                class IngredientConsum(BaseModel):
                    name: str
                    calculated_copies: int

                class MenuItem(BaseModel):
                    menu_name: str
                    ingredients_consumed: list[IngredientConsum]
                    seasonings: list[str]
                    recipe_steps: list[RecipeStep]

                class MenuResponse(BaseModel):
                    menus: list[MenuItem]

                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=MenuResponse,
                            temperature=0.2
                        ),
                    )
                    res_data = json.loads(response.text)
                    st.session_state['recommendations'] = res_data.get('menus', [])
                    st.session_state['selected_menu_idx'] = None  # 重置选择状态
                except Exception as e:
                    st.error(f"调用 Gemini 模型失败，请稍后重试。详情: {e}")

    # 菜单展示与选择逻辑
    if 'recommendations' in st.session_state and st.session_state['recommendations']:
        menus = st.session_state['recommendations']
        selected_idx = st.session_state.get('selected_menu_idx')
        
        # 状态一：用户未选择菜单时，并排显示 3 组推荐（显示菜单名、食材消耗量以及所需调味料）
        if selected_idx is None:
            cols_rec = st.columns(3)
            for idx, menu in enumerate(menus):
                with cols_rec[idx]:
                    st.subheader(f"推荐方案 {idx+1}")
                    st.markdown(f"### 🍳 {menu['menu_name']}")
                    
                    st.markdown("**🔹 拟消耗库存食材:**")
                    for ing in menu['ingredients_consumed']:
                        st.write(f"- {ing['name']}: {ing['calculated_copies']} 份")
                        
                    st.markdown("**🔹 所需调味料:**")
                    st.write(", ".join(menu['seasonings']))
                    
                    if st.button("📖 选择此菜单并查看菜谱", key=f"select_{idx}", use_container_width=True):
                        st.session_state['selected_menu_idx'] = idx
                        st.rerun()
        else:
            # 状态二：用户选择菜单后，隐藏没被选中的菜单，并显示选中菜单的菜谱
            chosen_menu = menus[selected_idx]
            
            st.success(f"已选择菜单：{chosen_menu['menu_name']}")
            
            # 菜谱分步骤逐行显示
            st.subheader("📖 20分钟快速烹饪菜谱 (已包含大体积食材切分处理)")
            for step in chosen_menu['recipe_steps']:
                st.write(f"**第 {step['step_num']} 步**: {step['description']}")
                
            st.markdown("---")
            act_cols = st.columns([3, 7])
            
            # 确认完成烹饪，自动扣除所需食材量，并立即写入文件
            if act_cols[0].button("✅ 确认做菜（自动扣除库存）", use_container_width=True):
                for ing in chosen_menu['ingredients_consumed']:
                    ing_name = ing['name']
                    deduct_qty = ing['calculated_copies']
                    
                    idx_list = df_inv[df_inv['食材'] == ing_name].index
                    if len(idx_list) > 0:
                        target_idx = idx_list[0]
                        current_qty = df_inv.at[target_idx, '份数']
                        if current_qty > deduct_qty:
                            df_inv.at[target_idx, '份数'] = current_qty - deduct_qty
                        else:
                            df_inv = df_inv.drop(target_idx)
                            
                save_inventory(df_inv)
                
                # 扣除成功后重置推荐状态与选择状态
                del st.session_state['recommendations']
                st.session_state['selected_menu_idx'] = None
                st.toast("库存已实时扣除并写入文件！")
                st.rerun()
                
            if act_cols[1].button("❌ 返回重新选择菜单", use_container_width=True):
                st.session_state['selected_menu_idx'] = None
                st.rerun()