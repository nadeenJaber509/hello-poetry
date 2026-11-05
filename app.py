
"""AI Learning Lab — A hands-on AI learning journey."""

import pandas as pd
import streamlit as st

from hello_poetry.data_utils import analyze_data, load_csv

st.set_page_config(
    page_title="AI Learning Lab",
    page_icon="🧠",
    layout="wide",
)


# --------------------------------------------------
# Design
# --------------------------------------------------

def apply_styles():
    """Apply the off-white and navy learning lab theme."""
    st.markdown(
        """
        <style>
        :root {
            --lab-navy: #122640;
            --lab-muted: #66758A;
            --lab-line: #DFE4E8;
            --lab-cream: #F8F7F3;
        }

        .stApp {
            background-color: var(--lab-cream);
        }

        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #122640;
        }

        /* Sidebar text and icons */
        [data-testid="stSidebar"] * {
            color: #EAF0F6;
        }

        /* Section headers */
        [data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {
            color: #AEC2D8;
            font-weight: 600;
        }

        /* Selected page */
        [data-testid="stSidebar"] [aria-current="page"] {
            background-color: #2E4865;
            border-radius: 8px;
        }

        /* Navigation hover */
        [data-testid="stSidebar"] a:hover {
            background-color: #253E59;
            border-radius: 8px;
        }
        /* Match Streamlit's top header to the page background */
[data-testid="stHeader"] {
    background-color: #F8F7F3;
}

/* Keep the main page background consistent */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #F8F7F3;
}
/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button {
    background-color: #294765;
    color: #FFFFFF !important;
    border: 1px solid #496680;
    border-radius: 8px;
}

[data-testid="stSidebar"] .stButton button:hover {
    background-color: #365777;
    color: #FFFFFF !important;
}

/* Chat input area */
[data-testid="stBottom"] {
    background-color: #F8F7F3;
}

[data-testid="stChatInput"] {
    background-color: #F8F7F3;
}

        .block-container {
            max-width: 1120px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }
        

        h1, h2, h3 {
            letter-spacing: -0.025em;
        }

        .lab-hero {
            background: linear-gradient(
                125deg, #122640 0%, #1B3855 100%
            );
            color: #FFFFFF;
            padding: 2.6rem;
            border-radius: 18px;
            margin-bottom: 1.7rem;
        }

        .lab-eyebrow {
            color: #B6C9DC;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }

        .lab-hero h1 {
            color: #FFFFFF !important;
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1.16;
            margin: 1rem 0;
        }

        .lab-hero p {
            color: #E1E8F0;
            font-size: 1rem;
            line-height: 1.7;
            max-width: 670px;
        }

        .lab-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 1.5rem;
        }

        .lab-tag {
            border: 1px solid #526A83;
            border-radius: 999px;
            padding: 5px 12px;
            font-size: 0.78rem;
            color: #E5EDF5;
        }

        .lab-kicker {
            color: var(--lab-muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }

        .lab-intro {
            color: var(--lab-muted);
            line-height: 1.7;
            margin-bottom: 1rem;
        }

        .lab-step-number {
            display: inline-block;
            padding: 0.65rem 0.85rem;
            background: #E7EBF0;
            border: 1px solid #D5DEE7;
            border-radius: 10px;
            color: #122640;
            font-size: 0.9rem;
            font-weight: 700;
        }

        .lab-step-desc {
            color: var(--lab-muted);
            font-size: 0.92rem;
            margin-top: 0.15rem;
            line-height: 1.5;
        }

        .lab-footer {
            color: var(--lab-muted);
            font-size: 0.85rem;
            line-height: 1.7;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--lab-line);
        }

        @media (max-width: 640px) {
            .lab-hero {
                padding: 1.6rem;
            }

            .block-container {
                padding-top: 1.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Sample dataset
# --------------------------------------------------

@st.cache_data
def get_sample_data():
    """Return a sample dataset for the Data Explorer."""
    return pd.DataFrame({
        "Name": ["Sara", "Ahmad", "Lina", "Omar", "Maya"],
        "Age": [21, 23, 22, 25, 24],
        "Score": [85, 92, 78, 88, 95],
        "Department": [
            "CS",
            "Engineering",
            "CS",
            "Business",
            "Engineering",
        ],
    })


# --------------------------------------------------
# Learning path content
# --------------------------------------------------

LEARNING_STEPS = [
    {
        "number": "01–02",
        "title": "Explore Your Data",
        "description": "Understand datasets, statistics, and visualizations.",
        "page": "data_exploration",
    },
    {
        "number": "03",
        "title": "Clean & Prepare Data",
        "description": "Handle missing values, duplicates, and data quality.",
        "page": "data_cleaning",
    },
    {
        "number": "04",
        "title": "Machine Learning",
        "description": "Train models and evaluate their predictions.",
        "page": "machine_learning",
    },
    {
        "number": "06",
        "title": "Language Model Classification",
        "description": "Explore text classification with LLMs.",
        "page": "llm_classification",
    },
    {
        "number": "07",
        "title": "Semantic Search",
        "description": "Find similar text using embeddings.",
        "page": "semantic_search",
    },
    {
        "number": "08",
        "title": "Document Retrieval",
        "description": "Understand chunking and vector database search.",
        "page": "document_search",
    },
    {
        "number": "09",
        "title": "Build a RAG Assistant",
        "description": "Combine document retrieval with generated answers.",
        "page": "rag_assistant",
    },
]


# --------------------------------------------------
# Home page
# --------------------------------------------------

def home_page():
    """Show the AI learning journey and topic links."""

    st.markdown(
        """
        <div class="lab-hero">
            <div class="lab-eyebrow">
                AI Learning Lab / Practical Learning Journey
            </div>
            <h1>From Data to<br>Intelligent Systems.</h1>
            <p>
                A growing collection of hands-on AI examples.
                Follow the learning path, try each application,
                and see how the concepts connect.
            </p>
            <div class="lab-tags">
                <span class="lab-tag">Python</span>
                <span class="lab-tag">Data Science</span>
                <span class="lab-tag">Machine Learning</span>
                <span class="lab-tag">LLMs</span>
                <span class="lab-tag">RAG</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="lab-kicker">THE LEARNING PATH</div>',
        unsafe_allow_html=True,
    )

    st.header("Learn step by step")

    st.markdown(
        '<p class="lab-intro">'
        "Start with data fundamentals, then explore machine "
        "learning, language models, and retrieval-based AI. "
        "Each task includes a working example you can try."
        "</p>",
        unsafe_allow_html=True,
    )

    for step in LEARNING_STEPS:
        with st.container(border=True):
            number_col, content_col = st.columns(
                [1, 7],
                gap="medium",
                vertical_alignment="center",
            )

            with number_col:
                st.markdown(
                    '<div class="lab-step-number">'
                    f'{step["number"]}'
                    "</div>",
                    unsafe_allow_html=True,
                )

            with content_col:
                st.page_link(
                    PAGE_LINKS[step["page"]],
                    label=step["title"],
                    icon=":material/arrow_forward:",
                )

                st.markdown(
                    '<div class="lab-step-desc">'
                    f'{step["description"]}'
                    "</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    st.subheader("How to use this lab")

    st.markdown(
        "1. **Choose a task** from the learning path or sidebar.\n"
        "2. **Try the example** using the provided data or your own input.\n"
        "3. **Explore the results** and notice how changing inputs "
        "affects the output."
    )

    st.divider()

    st.markdown(
        '<p class="lab-footer">'
        "Built during my AI training at ProGineer. "
        "This learning lab will continue to grow as I work "
        "on new exercises and experiments."
        "</p>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Data Explorer — original functionality preserved
# --------------------------------------------------

def data_explorer_page():
    """Display the Data Explorer page."""

    st.title("Task 1–2 | Data Exploration")

    st.write(
        "Data exploration is the first step in understanding "
        "a dataset before cleaning or model training."
    )

    with st.expander("What will you learn?"):
        st.markdown(
            "- Inspect dataset rows and columns.\n"
            "- Understand column data types.\n"
            "- Calculate summary statistics.\n"
            "- Find missing values.\n"
            "- Visualize numerical columns."
        )

    st.caption("Upload a CSV file or explore the sample dataset.")

    # Upload section
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = load_csv(uploaded_file)
            st.success("CSV file loaded successfully!")
        except (ValueError, UnicodeError) as error:
            st.error(str(error))
            st.stop()
    else:
        df = get_sample_data()
        st.info(
            "Using sample dataset. "
            "Upload a CSV to explore your own data."
        )

    # Dataset overview
    result = analyze_data(df)

    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)
    col1.metric("Rows", result["shape"][0])
    col2.metric("Columns", result["shape"][1])

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(), width="stretch")

    # Column types
    st.subheader("Column Types")
    st.dataframe(
        result["column_types"].rename("Data Type"),
        width="stretch",
    )

    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(
        result["statistics"],
        width="stretch",
    )

    # Missing values
    st.subheader("Missing Values")
    st.dataframe(
        result["missing_values"].rename("Missing Count"),
        width="stretch",
    )

    # Interactive charts
    st.sidebar.header("Chart Settings")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        selected_column = st.sidebar.selectbox(
            "Choose a column",
            numeric_columns,
        )

        chart_type = st.sidebar.selectbox(
            "Chart type",
            ["Histogram", "Line Chart", "Bar Chart"],
        )

        st.subheader(f"{selected_column} Visualization")

        if chart_type == "Histogram":
            st.bar_chart(
                df[selected_column]
                .dropna()
                .value_counts()
                .sort_index()
            )

        elif chart_type == "Line Chart":
            st.line_chart(df[selected_column])

        else:
            st.bar_chart(df[selected_column])

    else:
        st.warning(
            "No numeric columns available for visualization."
        )


# --------------------------------------------------
# Navigation — original task numbering preserved
# --------------------------------------------------

home = st.Page(
    home_page,
    title="Home",
    icon=":material/home:",
    default=True,
)

data_exploration = st.Page(
    data_explorer_page,
    title="Task 1–2 · Data Exploration",
    icon=":material/table_chart:",
)

data_cleaning = st.Page(
    "pages/1_Clean_Data.py",
    title="Task 3 · Data Cleaning",
    icon=":material/cleaning_services:",
)

machine_learning = st.Page(
    "pages/2_Train_Model.py",
    title="Task 4 · Train Model",
    icon=":material/model_training:",
)

llm_classification = st.Page(
    "pages/3_Classifier_Lab.py",
    title="Task 6 · LLM Classification",
    icon=":material/psychology:",
)

semantic_search = st.Page(
    "pages/4_Search.py",
    title="Task 7 · Semantic Search",
    icon=":material/search:",
)

document_search = st.Page(
    "pages/5_Docs_Search.py",
    title="Task 8 · Document Search",
    icon=":material/library_books:",
)

rag_assistant = st.Page(
    "pages/6_Poetry_Chat.py",
    title="Task 9 · RAG Assistant",
    icon=":material/smart_toy:",
)

PAGE_LINKS = {
    "data_exploration": data_exploration,
    "data_cleaning": data_cleaning,
    "machine_learning": machine_learning,
    "llm_classification": llm_classification,
    "semantic_search": semantic_search,
    "document_search": document_search,
    "rag_assistant": rag_assistant,
}

pages = {
    "Start Here": [home],
    "Data & Machine Learning": [
        data_exploration,
        data_cleaning,
        machine_learning,
    ],
    "Language AI & Retrieval": [
        llm_classification,
        semantic_search,
        document_search,
        rag_assistant,
    ],
}

apply_styles()
navigation = st.navigation(pages)
navigation.run()
