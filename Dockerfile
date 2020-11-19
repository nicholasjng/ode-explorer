FROM python:3.7.9

WORKDIR /ode_explorer

#copy file for caching leverage
COPY requirements.txt /ode_explorer/requirements.txt

# pip install the requirements
RUN pip install -r requirements.txt

#copy project files
COPY . .

# pip install ode-explorer
RUN pip install .