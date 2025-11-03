import streamlit as st

def mostrar_banner(titulo):
    logo = "https://raw.githubusercontent.com/Jorfanv/utilities/refs/heads/main/images/Logo%20gris.png"
    st.markdown("""
        <style>
            .banner {
                position: relative;
                background-color: #1F4E79;
                padding: 0.5vw;
                border-radius: 12px;
                margin-bottom: 2rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .banner h1 {
                color: white;
                font-size: min(2.5vw, 2.5vh);
                margin: 0 auto;
                font-family: Tahoma, sans-sarif;
                text-align: left;
                flex: 1;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .banner img {
                height: min(7vw, 8vh);
                max-height: 60px;
                margin-right: 1vw;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class='banner'>
            <h1>{titulo}</h1>
            <img src='{logo}' alt='Logo'>
        </div>
    """, unsafe_allow_html=True)
