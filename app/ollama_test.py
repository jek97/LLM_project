from typing import Tuple
import logging
import tempfile
import os
import re
import time

import click
from lxml import etree
import yaml

from gpt_interface import GPTInterface
from network_interface import NetworkInterface
from ollama_interface import OllamaInterface


class OllamaPlanner:
    def __init__(
        self,
        token_path: str,
        schema_path: str,
        farm_layout: str,
        image_path: str,
        max_retries: int,
        max_tokens: int, 
        temperature: float,
        log_directory: str,
        logger: logging.Logger,
        model: str,
        multimodal: bool,
    ):
        # logger instance
        self.logger: logging.Logger = logger
        # set schema and farm file paths
        self.schema_path: str = schema_path
        self.farm_layout: str = farm_layout
        self.image_path = image_path
        # logging GPT output folder
        self.log_directory: str = log_directory
        # max number of times that GPT can try and fix the mission plan
        self.max_retries: int = max_retries
        self.model_multimodal = multimodal
        
        if (model == "gpt"):
            self.logger.debug("Using GPT model")
            # init gpt interface
            self.gpt: GPTInterface = GPTInterface(self.logger, token_path, max_tokens, temperature)
            self.gpt.init_context(self.schema_path, self.farm_layout)
            self.gpt_flag = True # set flag for the correct initialization and future question

        else:
            self.logger.debug("Using Ollama ")
            # init ollama interface
            self.ollama: OllamaInterface = OllamaInterface(self.logger, max_tokens, temperature, model)
            self.ollama.init_context(self.schema_path, self.farm_layout)
            if self.model_multimodal:
                self.ollama.init_context_image(self.image_path)
            self.gpt_flag = False
        
    def configure_network(self, host: str, port: int) -> None:
        # network interface
        self.nic: NetworkInterface = NetworkInterface(self.logger, host, port)
        # start connection to ROS agent
        self.nic.init_socket()

    def parse_xml(self, mp_out: str) -> str:
        xml_response_0: str = mp_out.split("```xml\n")
        if len(xml_response_0 > 1):
            xml_response = xml_response_0[1]
            xml_response = xml_response.split("```")[0]
        else:
            xml_response = xml_response_0[0]

        return xml_response

    def write_out_xml(self, mp_out: str) -> str:
        # Create a temporary file in the specified directory
        with tempfile.NamedTemporaryFile(dir=self.log_directory, delete=False, mode="w") as temp_file:
            temp_file.write(mp_out)
            # name of temp file output
            temp_file_name = temp_file.name
        
        return temp_file_name

    def validate_output(self, xml_file: str) -> Tuple[bool, str]:
        try:
            # Parse the XSD file
            with open(self.schema_path, "rb") as schema_file:
                schema_root = etree.XML(schema_file.read())
            schema = etree.XMLSchema(schema_root)

            # Parse the XML file
            with open(xml_file, 'rb') as xml_file:
                xml_doc = etree.parse(xml_file)

            # Validate the XML file against the XSD schema
            schema.assertValid(xml_doc)
            self.logger.debug("XML input from ChatGPT has been validated...")
            return True, "XML is valid."

        except etree.XMLSchemaError as e:
            return False, "XML is invalid: " + str(e)
        except Exception as e:
            return False, "An error occurred: " + str(e)

    def run(self, quary):
        
        # ask user for their mission plan
        #mp_input: str = input("Enter the specifications for your mission plan: ")
        question_time_0 = time.time()
        if self.gpt_flag:
            mp_out: str = self.gpt.ask_gpt(quary, True)
        else:
            mp_out: str = self.ollama.ask_ollama(quary, True)
            
        question_time = time.time() - question_time_0 # get the time to obtain the first answer

        self.logger.debug(mp_out)
        mp_out = self.parse_xml(mp_out)
        output_path = self.write_out_xml(mp_out)
        asw_0 = output_path # saave the name of the file where i will store the answer
        self.logger.debug(f"GPT output written to {output_path}...")
        ret, e = self.validate_output(output_path)
        retry_times = [] # vector to store the time required for each adjustment
        asw_ret = []
        if not ret:
            retry: int = 0
            while not ret and retry < self.max_retries:
                self.logger.debug(f"Retrying after failed to validate GPT mission plan: {e}")
                ret_t_begin = time.time()
                if self.gpt_flag:
                    mp_out: str = self.gpt.ask_gpt(e, True)
                else:
                    mp_out: str = self.ollama.ask_ollama(e, True)

                ret_t_end = time.time() - ret_t_begin
                retry_times.append(ret_t_end)

                mp_out = self.parse_xml(mp_out)
                output_path = self.write_out_xml(mp_out)
                asw_ret.append(output_path)
                self.logger.debug(f"Temp GPT output written to {output_path}...")
                ret, e = self.validate_output(output_path)
                retry += 1
        # TODO: should we do this after every mission plan or leave them in context?
        self.ollama.reset_context()

        if not ret:
            self.logger.error("Unable to generate mission plan from your prompt...")
            succ_flag = False
        else:
            # TODO: send off mission plan to TCP client
            # self.nic.send_file(output_path)
            self.logger.debug("Successful mission plan generation...")      
            succ_flag = True  
            
        # TODO: decide how the reuse flow works
        # self.nic.close_socket()
        return question_time, asw_0, retry_times, asw_ret, succ_flag
    
    def release_resources(self):
        self.ollama.release_resources()
        self.logger.debug("resources relesed...")

def read_inputs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Regex to find all strings in double quotes
    quoted_strings = re.findall(r'"(.*?)"', text)
    
    return quoted_strings

def write_log(file_path, log):
    with open(file_path, "w") as file:
        file.write(log)

    return

@click.command()
@click.option(
    "--config",
    default="./app/config/localhost.yaml",
    help="YAML config file",
)
def main(config: str):
    log_file_path = "./app/gpt_outputs/ollama_outputs.txt"
    #models = ["gemma3:4b", "deepseek-r1:7b", "llama3.2:3b", "misral:7b",  "qwen2.5-coder:7b", "llava:7b"]
    #multimodals_models = [True, False, False, False, False, True]
    models = ["llava:7b"]
    multimodals_models = [True]
    inputs = read_inputs("./app/inputs.txt")
    temp = [0, 0.25, 0.5, 0.75, 1.0]
    for m, mode in zip(models, multimodals_models):
        for t in temp:
            with open(config, "r") as file:
                config_yaml: yaml.Node = yaml.safe_load(file)

            try:
                # configure logger
                logging.basicConfig(level=logging._nameToLevel[config_yaml["logging"]])
                logger: logging.Logger = logging.getLogger()

                mp: OllamaPlanner = OllamaPlanner(
                    config_yaml["token"],
                    config_yaml["schema"],
                    config_yaml["farm_layout"],
                    config_yaml["farm_image"],
                    config_yaml["max_retries"],
                    config_yaml["max_tokens"],
                    t,
                    config_yaml["log_directory"],
                    logger,
                    m,
                    mode
                )
                if m == "gpt":
                    mp.configure_network(config_yaml["host"], int(config_yaml["port"]))
                    logger.debug("Using GPT model")
            except yaml.YAMLError as exc:
                logger.error(f"Improper YAML config: {exc}")
            
            print("Using model", m)
            for i, inp in enumerate(inputs):
                print("input", inp)
                question_time, asw_0, retry_times, asw_ret, succ_flag = mp.run(inp)

                # prepere the log message
                log_msg = "input " + str(i) + ", model " + m + ", temp " + str(t) + ", question_time " + str(question_time) + ", answer " + asw_0
                for rt, asw_rt in zip(retry_times, asw_ret):
                    log_msg += ", ret " + str(rt) + ", answer " + asw_rt
                log_msg += ", succ " + str(succ_flag) + "\n"

                # log the message
                write_log(log_file_path, log_msg)

            mp.release_resources()
    return


if __name__ == "__main__":
    main()



