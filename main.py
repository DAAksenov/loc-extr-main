import streamlit as st
from Classes.streamlit_code import *
    
projects = {'Поиск экстремумов ФНП': page1,
            'Методы одномерной оптимизации': page2,
            'Градиентные методы многомерной оптимизации': page3,
            'Регрессия': page4,
            'Метод внутренней точки': page5,
            'Классификация': page6,
            'Метод отсекающих плоскостей': page7,
            'Методы стохастической оптимизации': page8
}

project = st.sidebar.selectbox("Выберите проект", options = projects.keys())
projects[project]()