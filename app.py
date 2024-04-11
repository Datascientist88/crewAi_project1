from crewai import Agent ,Task ,Crew ,Process
from crewai_tools import tool
from langchain_openai import ChatOpenAI
from crewai_tools.tools import FileReadTool
import os ,requests ,re ,mdpdf ,subprocess
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv
#Instantiate the llm
llm=ChatOpenAI(model="gpt-4-turbo-preview")
# Create the Tools 
file_read_tool=FileReadTool(
    file_path='template.md',
    descriptions='A tool to read a story Template File and understand the expected output'
)
@tool
def generateimage(chapter_content_and_character_details:str)->str:
    """
    Generates image for a given chapter number , chapter content , charcater details and detailed location using OPENAI image generation API ,saves it in the current folder,
    and returns the image path.
    """
    client=OpenAI()
    response=client.images.generate(
        model="dall-e-3",
        prompt=f"image is about:{chapter_content_and_character_details}. style:Illustration. Create an illustration",
        size="1024x1024",
        quality="standard",
        n=1
    )
    image_url=response.data[0].url
    words=chapter_content_and_character_details.split()[:5]
    safe_words=[re.sub(r'[^a-zA-Z0-9_]','',word) for word in words]
    filename="_".join(safe_words).lower()+".png"
    filepath=os.path.join(os.getcwd(),filename)
    # download the image fro URL
    image_response=requests.get(image_url)
    if image_response.status_code==200:
        with open(filepath,'wb') as file:
            file.write(image_response.content)
    else:
        print("Failed to download the image")
    return filepath
@tool
def convertmarkdowntopdf(markdownfile_name:str)->str:
    """
    converts markdown file to a PDF document using the mdpdf command line application.
    Args:
        markdownfile_name(str):Path to the input markdown file.
    Return:
        str:path to the output PDF File.

    """
    output_pdf=markdownfile_name.replace('.md','.pdf')
    subprocess.run(['mdpf',markdownfile_name,'-o',output_pdf])
    return output_pdf
# create 5 Agents
story_outliner=Agent(
    role='Story Outliner',
    goal="Develop an Outline for a children\'s storybook about Animals including chapter titles and characters.",
    backstory="An imaginative creator who lays the foundation of captivating stories for children.",
    verbose=True,
    llm=llm,
    allow_delegation=False
)
story_writer=Agent(
    role='Story Writer',
    goal="Write the full content of the story for all 5 chapters,each chapter 100 words ,weaving together.",
    backstory="A talented Storyteller who brings to life the world and characters outlined crafting engaging story.",
    verbose=True,
    llm=llm,
    allow_delegation=False
)
image_generator=Agent(
    role='Image Generator',
    goal="Generate one image per chapter provided by the story outliner. start with chatper number, character .",
    backstory="A creative AI specialized in visual storytelling , bringing each chapter to life through imagination.",
    verbose=True,
    llm=llm,
    tool=[generateimage],
    allow_delegation=False
)
content_formatter=Agent(
    role='Content Formatter',
    goal="Format the written story content in a markdown including images at the begining of each chapter.",
    backstory="A meticulous formatter who enhances the readability and presentation of the storybook.",
    verbose=True,
    llm=llm,
    tool=[file_read_tool],
    allow_delegation=False
)
markdown_to_pdf_creator=Agent(
    role='PDF Converter',
    goal="Convert the markdown file into PDF Document .story.md isthe markdown file name .",
    backstory="An efficient converter that transforms Markdown files into professionally formatted PDF document.",
    verbose=True,
    llm=llm,
    tool=[convertmarkdowntopdf],
    allow_delegation=False
)
# Create 5 Tasks 
task_outline=Task(
    description="Create an outline for the children\'s storybook about animals detailing chapter titles and character descriptions for 5 chapters.",
    agent=story_outliner,
    expected_output="A structured Outline document containing 5 chapter titles with detailed character descriptions and the main plot points for each chapter."

)
task_write=Task(
    description="Using the outline provided , write the full story content for all chapters , ensuring a cohesive flow of the story  .",
    agent=story_writer,
    expected_output="A Complete manuscript of the children\'s storybook 5 chapters  ."

)
task_image_generate=Task(
    description="Generate 5 images that cpature the essence of the children\'s storybook about Animals  .",
    agent=image_generator,
    expected_output="A ditigal image file that visually represents the overarching theme of the children\'s storybook ."

)
task_format_content=Task(
    description="Format story content in  markdown ,including an image  at the begining of each chapter.",
    agent=content_formatter,
    expected_output="The entire storybook is formatted in markdown with each containing the title followed by the image and below it the story belonging to that chapter.",
    context=[task_write,task_image_generate],
    output_file="story.md"

)
task_markdown_to_pdf=Task(
    description="Convert a markdown file to PDF document ensuring preservation of formatting and Structure .",
    agent=markdown_to_pdf_creator,
    expected_output="A PDF file generated from  the markdown  accurately reflecting the content  ."

)
crew=Crew(
    agents=[story_outliner,story_writer,image_generator,content_formatter,markdown_to_pdf_creator],
    tasks=[task_outline,task_write,task_image_generate,task_format_content,task_markdown_to_pdf],
    verbose=True,
    process=Process.sequential
)

result=crew.kickoff()
print(result)
