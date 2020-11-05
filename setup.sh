mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"nkosilati23@yahoo.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\