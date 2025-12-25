# 多维度追踪看板应用

一个用于追踪多维度数据的个人看板应用，支持每日记录、雷达图可视化和周统计功能。

## 功能特性

-   📊 **双看板支持**：同时管理"边界感六维"和"2026 找工作八维"两个看板
-   📝 **每日记录**：为每个维度记录 0-10 的数值
-   📈 **雷达图可视化**：直观展示本周各维度的平均值
-   📊 **周统计对比**：显示本周均值与上周的变化趋势
-   🗂️ **历史记录管理**：查看和删除历史记录

## 技术栈

### 后端

-   **FastAPI**：现代化的 Python Web 框架
-   **SQLAlchemy**：ORM 数据库操作
-   **SQLite**：轻量级数据库
-   **Pydantic**：数据验证

### 前端

-   **原生 HTML/CSS/JavaScript**：无框架依赖
-   **Chart.js**：雷达图可视化
-   **响应式设计**：适配不同屏幕尺寸

## 项目结构

```
LevelUp/
├── main.py              # 后端 API 和数据库模型
├── static/
│   └── index.html      # 前端页面
├── kanban.db           # SQLite 数据库文件（自动生成）
├── pyproject.toml      # 项目依赖配置
└── README.md           # 项目说明
```

## 数据库设计

### Board（看板表）

-   `id`: 主键
-   `name`: 看板名称（如"边界感六维"）
-   `dimensions`: JSON 格式的维度列表

### DailyRecord（每日记录表）

-   `id`: 主键
-   `board_id`: 关联看板 ID
-   `record_date`: 记录日期
-   `values`: JSON 格式的维度数值（维度名 -> 数值）

## API 接口

### 看板相关

-   `GET /boards` - 获取所有看板
-   `GET /boards/{board_id}` - 获取单个看板信息
-   `GET /boards/{board_id}/weekly-stats` - 获取每周统计数据

### 记录相关

-   `GET /records` - 获取记录列表（支持筛选）
-   `POST /records` - 创建或更新每日记录
-   `GET /records/{record_id}` - 获取单条记录
-   `DELETE /records/{record_id}` - 删除记录

## 快速开始

### 本地开发

1. 安装依赖：

```bash
uv sync
```

2. 运行应用：

```bash
python main.py
```

3. 访问应用：`http://localhost:8000`

4. 手机访问（同一 WiFi）：`http://电脑IP:8000`

---

## 🚀 部署到 Railway（免费云平台）

### 一键部署步骤：

1. **推送代码到 GitHub**

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/你的用户名/LevelUp.git
git push -u origin main
```

2. **连接 Railway**

    - 访问 [railway.app](https://railway.app)
    - 用 GitHub 账号登录
    - 点击 "New Project" → "Deploy from GitHub repo"
    - 选择你的 LevelUp 仓库
    - Railway 会自动检测 Python 项目并部署

3. **获取访问地址**

    - 部署完成后，点击 "Settings" → "Generate Domain"
    - 获得类似 `levelup-xxx.up.railway.app` 的地址

4. **手机安装 PWA**
    - 手机浏览器访问你的 Railway 地址
    - 点击浏览器菜单 → "添加到主屏幕"
    - 像原生 App 一样使用！

### 数据同步

部署后，手机和电脑访问同一个地址，数据自动同步！

---

## 📱 PWA 支持

本应用支持 PWA（渐进式 Web 应用）：

-   ✅ 可添加到手机主屏幕
-   ✅ 全屏运行，无浏览器地址栏
-   ✅ 自定义图标和启动画面
-   ✅ 基本离线缓存支持

## 使用说明

1. **记录数据**：选择日期，为每个维度输入 0-10 的数值，点击"保存记录"
2. **查看统计**：页面自动显示本周各维度的平均值和与上周的对比
3. **查看图表**：雷达图实时展示本周各维度的数据分布
4. **管理记录**：在历史记录区域可以查看和删除过往记录

## 学习要点

### 后端架构

-   RESTful API 设计
-   数据库模型设计（ORM）
-   数据验证（Pydantic）
-   统计计算逻辑

### 前端架构

-   原生 JavaScript 异步编程
-   Chart.js 图表库使用
-   响应式 CSS Grid 布局
-   表单处理和用户交互

## 后续可以改进的方向

1. **数据导出**：支持导出为 CSV 或 Excel
2. **数据可视化增强**：添加折线图显示趋势
3. **用户系统**：支持多用户和登录
4. **数据备份**：自动备份功能
5. **移动端优化**：更好的移动端体验
6. **数据筛选**：按日期范围筛选记录
7. **目标设定**：为每个维度设定目标值
