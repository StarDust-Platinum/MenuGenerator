import os
import json
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---- 1. 基础路径与常量配置 ----
DATA_DIR = "data"
INVENTORY_PATH = os.path.join(DATA_DIR, "inventory.csv")
CATEGORY_PATH = os.path.join(DATA_DIR, "category.csv")
MENU_PATH = os.path.join(DATA_DIR, "menu.json")

st.set_page_config(page_title="食材库存管理与菜单推荐", layout="wide")

# ---- 2. 数据初始化与加载逻辑 ----
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 加载食材基础标准规格定义
if os.path.exists(CATEGORY_PATH):
    df_cat = pd.read_csv(CATEGORY_PATH)
else:
    st.error(f"未找到必要的基础定义文件: {CATEGORY_PATH}，请先创建。")
    st.stop()

# 逻辑：inventory.csv 不存在时，创建文件和表头但不添加记录
if not os.path.exists(INVENTORY_PATH):
    df_empty = pd.DataFrame(columns=["食材", "单位质量", "份数"])
    df_empty.to_csv(INVENTORY_PATH, index=False, encoding='utf-8')

# 逻辑：menu.json 不存在时，创建文件但不添加内容
if not os.path.exists(MENU_PATH):
    with open(MENU_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

def load_inventory():
    df = pd.read_csv(INVENTORY_PATH, encoding='utf-8')
    if df.empty:
        return df
    # 严格按照数据规格定义进行数据转换
    df["单位质量"] = df["单位质量"].astype(int)
    df["份数"] = df["份数"].astype(int)
    # 逻辑：按数据规格定义中的顺序排序 (依据 category.csv 的行序)
    cat_order = df_cat["食材"].tolist()
    df["食材"] = pd.Categorical(df["食材"], categories=cat_order, ordered=True)
    df = df.dropna().sort_values("食材").reset_index(drop=True)
    df["食材"] = df["食材"].astype(str)
    return df

def save_inventory(df):
    # 逻辑：数量为 0 的记录自动删除
    df = df[df["份数"] > 0]
    df.to_csv(INVENTORY_PATH, index=False, encoding='utf-8')

def load_menu():
    try:
        with open(MENU_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_menu(menu_data):
    with open(MENU_PATH, "w", encoding="utf-8") as f:
        json.dump(menu_data, f, ensure_ascii=False, indent=4)

# ---- 3. Gemini 2.5 Flash 结构化 Pydantic Schema 定义 ----
class IngredientsConsumed(BaseModel):
    name: str = Field(description="食材名称，必须与库存食材名精确一致")
    calculated_copies: int = Field(description="消耗的份数")

class RecipeStep(BaseModel):
    step_num: int = Field(description="步骤序号，自1开始的整数")
    description: str = Field(description="当前步骤的烹饪或预处理详细描述")

class MenuItem(BaseModel):
    menu_name: str = Field(description="菜名")
    ingredients_consumed: list[IngredientsConsumed] = Field(description="食材消耗量列表")
    seasonings: list[str] = Field(description="所需调味料列表")
    recipe_steps: list[RecipeStep] = Field(description="分步骤菜谱")

def generate_menu_via_gemini(inventory_df):
    if inventory_df.empty:
        st.warning("当前无食材库存，请先添加食材后再生成推荐。")
        return None

    inv_list = []
    for _, row in inventory_df.iterrows():
        inv_list.append(f"- {row['食材']}: {row['份数']} 份 (每份 {row['单位质量']}g)")
    inv_context = "\n".join(inv_list)

    prompt = f"""
你是一名高级家庭营养师。请根据以下现有的食材库存，推荐 3 组完全独立的菜单。

【当前可用库存】：
{inv_context}

【硬性约束条件】：
1. 必须且仅推荐 3 组菜单。
2. 菜单所用食材量（calculated_copies）必须在上述库存量以内，绝对不能推荐库存不足以制作的菜单。
3. 包括食材预处理在内的总烹饪时间必须控制在 20 分钟以内。
4. 菜单必须营养均衡，口味清淡，完美满足 2 名成人和 1 名 18 个月儿童的膳食所需量（食物需精细软烂、低盐无刺激）。
"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("未在环境中检测到 GEMINI_API_KEY，请检查环境变量配置。")
        return None

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[MenuItem],
                temperature=0.2
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Gemini 2.5 Flash API 请求异常: {e}")
        return None

# ---- 4. 初始化数据状态 ----
df_inv = load_inventory()
current_menus = load_menu()

if "active_menu_idx" not in st.session_state:
    st.session_state.active_menu_idx = None


# ==================== 全局 UI 顺序 1：食材库存数据与数据操作区域 ====================
st.title("🥬 食材库存管理与智能菜单推荐系统")
st.header("食材库存数据与数据操作区域")

if df_inv.empty:
    st.info("当前库存暂无记录，请在下方添加新食材记录。")
else:
    # 纯净局部 CSS 注入：完全不影响字色。清除边框和分割线线，实现纯净的原生卡片行交替背景
    st.html("""
        <style>
        /* 1. 核心需求：彻底移除表格中各行之间的分割线与多余容器边框 */
        [data-testid="stVerticalBlockBorderDiv"] {
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            gap: 0 !important;
        }
        /* 2. 核心需求：利用原生选择器，对表格行大容器精准应用交替背景色（偶数行添加半透明遮罩） */
        /* 采用 rgba 透明度，浅色模式表现为优雅淡灰，深色模式表现为微亮暗灰，双模式完美适配且绝不吃字 */
        .inventory-table-container > div[data-testid="stVerticalBlock"] > div:nth-of-type(even) {
            background-color: rgba(128, 128, 128, 0.08) !important;
            padding: 8px 12px !important;
            border-radius: 6px !important;
        }
        .inventory-table-container > div[data-testid="stVerticalBlock"] > div:nth-of-type(odd) {
            padding: 8px 12px !important;
            border-radius: 6px !important;
        }
        </style>
    """)

    # 渲染标准表格头部 (原生 Markdown 颜色自适应)
    h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns([3, 2, 2, 2, 1])
    h_col1.markdown("**食材**")
    h_col2.markdown("**单位质量**")
    h_col3.markdown("**当前份数**")
    h_col4.markdown("**编辑数量**")
    h_col5.markdown("**操作**")
    st.markdown("---")

    # 外层包裹定位类容器，配合上方的高级 CSS，使交替色及去线逻辑只生效于库存表格中
    with st.container(key="inventory-table-container"):
        for idx, row in df_inv.iterrows():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
                
                # 抛弃 HTML 拼接，利用原生组件在浅色下渲染黑字，深色下渲染白字
                col1.markdown(row['食材'])
                col2.markdown(f"{row['单位质量']} g")
                col3.markdown(f"**{row['份数']}** 份")
                
                # 功能逻辑：用户可以编辑已存在记录的数量
                new_copies = col4.number_input(
                    "编辑数量", 
                    min_value=0, 
                    value=int(row["份数"]), 
                    key=f"edit_input_{row['食材']}_{idx}", 
                    label_visibility="collapsed"
                )
                if new_copies != row["份数"]:
                    df_inv.at[idx, "份数"] = new_copies
                    save_inventory(df_inv) # 逻辑：更改库存数据后需立即写入至 inventory.csv
                    st.rerun()
                    
                # 功能逻辑：每一行记录右方有删除按钮
                if col5.button("🗑️ 删除", key=f"btn_del_{row['食材']}_{idx}", use_container_width=True):
                    df_inv.at[idx, "份数"] = 0
                    save_inventory(df_inv) # 逻辑：数量为 0 的记录自动删除并更新
                    st.rerun()

# 局部 UI：表格下方有新建记录 UI
st.markdown("#### ➕ 新建库存记录")
# 功能逻辑：写入后自动清空表单内容 (clear_on_submit=True)
with st.form("new_ingredient_form", clear_on_submit=True):
    existing_ingredients = df_inv["食材"].tolist() if not df_inv.empty else []
    # 功能逻辑：新添加的记录必须在 category.csv 定义之中，并过滤防止产生同名冲突
    allowed_choices = df_cat[~df_cat["食材"].isin(existing_ingredients)]["食材"].tolist()
    
    if not allowed_choices:
        st.info("系统定义文件中的所有基础食材已全部在库存中。")
        form_submit = None
    else:
        chosen_name = st.selectbox("选择添加食材", options=allowed_choices)
        chosen_copies = st.number_input("份数", min_value=1, value=1, step=1)
        form_submit = st.form_submit_button("写入库存")
        
    if form_submit and chosen_name:
        std_weight = int(df_cat[df_cat["食材"] == chosen_name]["单位质量/g"].values[0])
        new_row = pd.DataFrame([{"食材": chosen_name, "单位质量": std_weight, "份数": chosen_copies}])
        df_inv = pd.concat([df_inv, new_row], ignore_index=True)
        save_inventory(df_inv) # 逻辑：更改库存数据后需立即写入至 inventory.csv
        st.rerun()


# ==================== 全局 UI 顺序 2：菜单推荐与菜谱显示区域 ====================
st.write("---")
st.header("菜单推荐与菜谱显示区域")

# 局部 UI：最上方提供一个生成菜单的按钮
if st.button("🤖 智能生成菜单 (Gemini 2.5 Flash)", type="primary"):
    with st.spinner("AI 正在深度检索库存并定制20分钟清淡幼儿营养家庭餐..."):
        recommended_results = generate_menu_via_gemini(df_inv)
        if recommended_results:
            save_menu(recommended_results)
            st.session_state.active_menu_idx = None  # 重置单选索引状态
            st.success("成功推荐 3 组独立菜单，已保存至 menu.json！")
            st.rerun()

# 功能逻辑：app 运行时加载 menu.json 并显示
if not current_menus:
    st.info("当前无菜单缓存。请确保库存有食材，并点击上方按钮自动生成。")
else:
    # 局部 UI分支一：用户选择菜单后，隐藏没被选中的菜单信息，保留选中的菜单信息并显示菜谱
    if st.session_state.active_menu_idx is not None:
        selected_idx = st.session_state.active_menu_idx
        menu_item = current_menus[selected_idx]
        
        st.subheader(f"🍳 选定推荐菜单：{menu_item['menu_name']}")
        
        col_view1, col_view2 = st.columns(2)
        with col_view1:
            st.markdown("**📉 食材消耗量：**")
            for ing in menu_item["ingredients_consumed"]:
                st.write(f"- {ing['name']}: {ing['calculated_copies']} 份")
        with col_view2:
            st.markdown("**🧂 所需调味料：**")
            st.write("、".join(menu_item["seasonings"]))
            
        st.markdown("---")
        # 局部 UI / 菜单规格：菜谱需分步骤逐行显示
        st.markdown("**📋 详细烹饪菜谱步骤（含预处理20分钟内）：**")
        for step in menu_item["recipe_steps"]:
            st.write(f"**第 {step['step_num']} 步**: {step['description']}")
            
        st.markdown("---")
        # 局部 UI：菜谱下方显示制作完成按钮
        if st.button("✅ 制作完成", type="primary", use_container_width=True):
            # 功能逻辑：用户选择菜单并点击制作完成后，自动扣除所需食材份数并写入 inventory.csv
            for ing in menu_item["ingredients_consumed"]:
                df_inv.loc[df_inv["食材"] == ing["name"], "份数"] -= ing["calculated_copies"]
            
            # 功能逻辑：数量为 0 的记录自动删除由 save_inventory 内部过滤统一持久化
            save_inventory(df_inv)
            st.session_state.active_menu_idx = None  # 退出菜谱视图，解除锁定返回主列表
            st.success(f"《{menu_item['menu_name']}》制作完成！相应食材库存份数已自动完成扣除。")
            st.rerun()
            
        if st.button("↩️ 取消并返回选择列表", use_container_width=True):
            st.session_state.active_menu_idx = None
            st.rerun()

    # 局部 UI分支二：初始状态，水平并排对齐显示 3 组推荐菜单，且最下方选择按钮需水平对齐
    else:
        st.markdown("### 💡 AI 为您推荐的餐单（请任选一组）：")
        menu_cols = st.columns(3)
        
        for idx, menu_item in enumerate(current_menus):
            if idx >= len(menu_cols):
                break  # 约束一次只显示3组菜单
                
            with menu_cols[idx]:
                st.markdown(f"### 🍱 {menu_item['menu_name']}")
                st.markdown("**食材消耗：**")
                stock_is_sufficient = True
                for ing in menu_item["ingredients_consumed"]:
                    match_row = df_inv[df_inv["食材"] == ing["name"]]
                    current_stock = int(match_row["份数"].values[0]) if not match_row.empty else 0
                    
                    if current_stock >= ing["calculated_copies"]:
                        st.write(f"✅ {ing['name']} × {ing['calculated_copies']} 份")
                    else:
                        st.write(f"❌ {ing['name']} × {ing['calculated_copies']} 份 *(现存量不足！当前库存：{current_stock} 份)*")
                        stock_is_sufficient = False
                        
                st.markdown("**所需调味料：**")
                st.write(", ".join(menu_item["seasonings"]))
                st.markdown("---")
                
        # 关键局部 UI 需求：为了实现“多个菜单的按钮需水平对齐”，把选择按钮脱离出文字内容区，在外部统一由同一排 columns 渲染
        btn_cols = st.columns(3)
        for idx, menu_item in enumerate(current_menus):
            if idx >= len(btn_cols):
                break
                
            with btn_cols[idx]:
                stock_is_sufficient = True
                for ing in menu_item["ingredients_consumed"]:
                    match_row = df_inv[df_inv["食材"] == ing["name"]]
                    current_stock = int(match_row["份数"].values[0]) if not match_row.empty else 0
                    if current_stock < ing["calculated_copies"]:
                        stock_is_sufficient = False
                
                if stock_is_sufficient:
                    if st.button(f"选择菜单: {menu_item['menu_name']}", key=f"select_btn_{idx}", use_container_width=True):
                        st.session_state.active_menu_idx = idx
                        st.rerun()
                else:
                    st.button("食材库存不足，无法选择", key=f"select_btn_{idx}", disabled=True, use_container_width=True)