# Use Python as the base image
FROM python:3.10-slim


# Set environment variables to prevent Python from writing .pyc files
# and buffering stdout and stderr.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app
RUN apt-get update
RUN apt-get install git -y
RUN pip install --upgrade pip
RUN pip install spacy
RUN python -m spacy download en_core_web_sm
RUN pip install beautifulsoup4
RUN pip install textblob





COPY requirements.txt /app/

# Install the dependencies without using cache

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall

#RUN cd /app
#RUN git clone https://github.com/ramsima/Fake_News_Detection.git



# Copy the entire project to the working directory
COPY . /app/



# Expose the port that the app runs on
EXPOSE 8000

# Set the entry point for the container
ENTRYPOINT ["python3"]

# Default command when the container starts

CMD ["manage.py","runserver", "0.0.0.0:8000"]