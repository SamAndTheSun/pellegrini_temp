'''
functions that manipulate the web browser for the purposes of data analysis
'''
from selenium import webdriver
from bs4 import BeautifulSoup

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select

from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

import os
import requests

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import string
import time

from liftover import get_lifter


def get_great(probe_data):

    '''
    find gene association data by position, for mm10
        param probe_data: dataframe to use for analysis, requires chr_mm10, pos_mm10, and end_mm10 columns for position, with indices being probe names, df

        return: dataframe with added gene assocations by probe, df
    '''

    #Establish settings
    options = Options()

    #create list for storing probe:gene combinations
    gene_by_probes = []

    #leaves browser open, useful for troubleshooting, turn off headless for this to work
    #options.add_experimental_option('detach', True)

    options.add_argument('--ignore-ssl-errors=yes') #ignore insecure warning
    options.add_argument('--ignore-certificate-errors')
    options.add_argument("--headless") #makes it so that the browser doesn't open

    #prep BED data
    bed_format = pd.DataFrame({'chr_mm10' : probe_data['chr_mm10'], 
                               'pos_mm10' : probe_data['pos_mm10'].astype(int),
                               'end_mm10' : probe_data['end_mm10'].astype(int), 
                               'index' : probe_data.index})
    
    #if too much data for GREAT to work properly, split the data
    n_iter = (bed_format.shape[0])//100000
    bed_n = {}
    if n_iter > 1:
        n = 0
        while n <= n_iter:
            bed_n[n] = bed_format.iloc[(n)*100000:(n+1)*100000]
            n+=1
    else:
        bed_n[0] = bed_format

    for key in bed_n.keys():

        #establish driver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                                options=options)
        driver.get('https://great.stanford.edu/great/public/html/')
        
        #set assembly to mm10
        set_assembly = driver.find_element(By.ID, 'mm10')
        set_assembly.click()

        #select 'BED data'
        use_input = driver.find_element(By.ID, 'fgChoiceData')
        use_input.click()

        #put BED data into text box
        bed_data_string = bed_n[key].to_csv(index=False, header=None, sep='\t')
        driver.execute_script("arguments[0].value = arguments[1];", driver.find_element(By.NAME, "fgData"), bed_data_string)

        #submit data
        submit = driver.find_element(By.ID, 'submit_button')
        submit.click()

        #wait for the table
        try:
            newelem = WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.ID, 'job_description_container')))
        except TimeoutException:
            print('Error: Loading exceeded 1 minute')
        
        #expand the table
        driver.execute_script("document.getElementById('job_description_container').style.display = 'block';")

        #show the gene associations
        show_table_button = driver.find_element(By.LINK_TEXT, 'View all genomic region-gene associations.')
        show_table_button.click()

        #Load the gene association data
        driver.switch_to.window(driver.window_handles[1]) #focus driver on newly opened tab
        soup = BeautifulSoup(driver.page_source, 'lxml')
        
        #Find the relevant gene table
        tables = soup.find_all('table', class_='gSubTable')

        #Find all gene names and positions
        gene_tags = tables[0].find_all('td')

        #prepare to create list of genes from table
        gene_list = []

        #Extract gene names / positions by probe
        for tag in gene_tags:
            if '_' in tag.text:
                gene_by_probes.append(gene_list)
                gene_list = []
            else:
                gene_list.append(tag.text)

        del(gene_by_probes[(key*100000)]) #remove empty list []
        gene_by_probes.append(gene_list) #add genes for last probe
        
    probe_data['associated_genes'] = gene_by_probes #add gene list to dataframe
    driver.quit()

    return probe_data

def get_cistrome(probe_data, fig_w=2800, fig_h=3000, check_pval=True, top_10k=False):

    '''
    generates cistrome plots using the inputted data

        param probe_data: dataset to be used, df, must contain probe position data, df
        param fig_w: figure width for cistrome subplots, int
        param fig_h: figure height for cistrome subplots, int
        check_pval: if the code should check for columns with '_pval' to determine how to rank probes,
            ranking with the lowest pval being moved to the top, etc., bool
        top_10k: if the code should utilize the top 10k probes from the dataset, or use the default 1k, bool

        return: none
    '''

    #Establish settings
    options = Options()

    #leaves browser open, useful for troubleshooting, turn off headless for this to work
    #options.add_experimental_option('detach', True)

    options.add_argument('--ignore-ssl-errors=yes') #ignore insecure warning
    options.add_argument('--ignore-certificate-errors')
    options.add_argument("--headless") #makes it so that the browser doesn't open, Cistrome is grumpy so this is necesary

    #establish driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                              options=options)

    #iterate through data and produce graphs
    images = []
    labels = []
    n_traits = 0
    for column in probe_data.columns:
        if (('_pval' in str(column)) or (check_pval==False)):

            #prep data for analysis
            bed_data = probe_data[[column,'chr_mm10', 'pos_mm10']] #get desired trait and position information
            bed_data = bed_data[bed_data[column].notna()] #keep defined values
            bed_data = bed_data.sort_values(by=column) #sort for desired order

            if top_10k: bed_data = bed_data.head(10000) #top 10k
            else: bed_data = bed_data.head(1000) #top 1k

            #put this into actual bed format
            bed_data['end_mm10'] = bed_data['pos_mm10']+2
            bed_data = bed_data.drop(columns=[column])

            if bed_data.empty:
                print(f'No valid probes, skipping trait {column}')
                continue

            #create target file
            bed_data.to_csv('temp.txt', sep='\t', index=False, header=False)

            #re-establish driver
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                            options=options)

            #direct driver to Cistrome
            driver.get('http://dbtoolkit.cistrome.org')

            #change assembly to mm10
            select = Select(driver.find_element(By.XPATH, '/html/body/div[4]/div/div/div/div[2]/div/form/div[1]/div[2]/div[1]/div/select'))
            select.select_by_visible_text('Mouse mm10')

            #if you want the top 10k probes to be used, select the top 10k option
            if top_10k:
                select = Select(driver.find_element(By.XPATH, '/html/body/div[4]/div/div/div/div[2]/div/form/div[1]/div[4]/div[1]/div/select'))
                select.select_by_visible_text('Top 10k peaks according to peak enrichment')
            else:
                pass

            #select 'choose file' button
            select_file = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, 'peak')))

            #upload probe set
            select_file.send_keys(str(os.getcwd()) + '/temp.txt')

            #press submit button
            submit_button = driver.find_element(By.XPATH, '/html/body/div[4]/div/div/div/div[2]/div/form/div[2]/div[2]/input')
            submit_button.click()

            #select figures
            show_plot = driver.find_element(By.LINK_TEXT, 'Result in figure')
            show_plot.click()

            #select iframe
            iframe = driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div/div[2]/div[3]/div/div[2]/iframe')
            driver.switch_to.frame(iframe)

            #select static plot
            static = driver.find_element(By.ID, 'staticli')
            static.click()

            #make plot larger
            plot_full = driver.find_element(By.LINK_TEXT, 'Download')
            plot_full.click()
            driver.switch_to.window(driver.window_handles[1]) #focus driver on newly opened tab

            #save plot
            png_element = driver.find_element(By.TAG_NAME, 'img')
            png_url = png_element.get_attribute('src')
            response = requests.get(png_url)

            #save image to working directory
            if check_pval:
                file_name = f'{column[:-5]}_temp_img.png'
            else:
                file_name = f'{column}_temp_img.png'
            try:
                with open(file_name, 'wb') as file:
                    file.write(response.content)
            except FileNotFoundError:
                if '/' in file_name:
                    file_name = file_name.replace('/', '_')
                    with open(file_name, 'wb') as file:
                        file.write(response.content)
                else:
                    print('Error: illegal character in trait name, modify exception accordingly')
                    pass
            path = os.path.join(os.getcwd(), file_name)
        
            images.append(path)
            labels.append(file_name[:-13])

            n_traits+=1

            if check_pval == False:
                break
        else:
            pass


    #establish subplots 
    px = 1/plt.rcParams['figure.dpi']
    if n_traits % 2 == 0:
        fig, axs = plt.subplots(n_traits, 2, figsize=(fig_w*px, fig_h*px))
    else:
        fig, axs = plt.subplots((n_traits//2)+1, 2, figsize=(fig_w*px, fig_h*px))

    #display images on subplots
    for i, ax in enumerate(axs.flat):
        try:
            plot = plt.imread(images[i])
            ax.imshow(plot, extent=[0, 1, 0, 1], aspect='auto')
            ax.axis('off')
            ax.set_title(labels[i], size=20)
        except IndexError:
            fig.delaxes(ax)
            pass

        #add subplot indicator text
        ax.text(0.1, 1.1, string.ascii_uppercase[i], transform=ax.transAxes, 
            size=25, weight='bold')

    #remove temp file
    os.remove('temp.txt')

    #remove images
    for image in images:
        os.remove(image)

    driver.quit()
    return

def get_pos(probe_data, ref_data, drop_undef=True):
    '''
    gets genetic information from illumina reference table, as provided, and prints out BED information

        param probe_data: data from which assessed probe names are derived, requires index to be probe names, df
        param ref_data: data indicating the position of probes, df
        param drop_undef: if True, drop undefined probes

        return: probe_data with chromosome and position added as columns, df
    '''

    chr_mm39 = []
    pos_mm39 = []

    chr_mm10 = []
    pos_mm10 = []

    converter = get_lifter('mm39', 'mm10', one_based=True)
    n_probes = probe_data.shape[0]

    n = 0
    while n < n_probes:
        bed_data = ref_data.loc[probe_data.index[n]]
        chr = bed_data.loc['chr']
        pos = bed_data.loc['start']

        chr_mm39.append(chr)
        pos_mm39.append(int(pos))

        try:
            chr_mm10.append(converter[chr][pos][0][0])
            pos_mm10.append(int(converter[chr][pos][0][1]))
        except (IndexError, KeyError): #IndexError: No conversion, KeyError: Initial file gives nan
            chr_mm10.append(np.nan)
            pos_mm10.append(np.nan)

        n+=1

    probe_data['chr_mm39'] = chr_mm39
    probe_data['pos_mm39'] = pos_mm39
    probe_data['end_mm39'] = probe_data['pos_mm39']+2

    probe_data['chr_mm10'] = chr_mm10
    probe_data['pos_mm10'] = pos_mm10
    probe_data['end_mm10'] = probe_data['pos_mm10']+2

    if drop_undef:
        probe_data = probe_data.dropna()
    else:
        pass

    return probe_data

def screenshot(driver):
    '''
    generates a screenshot corresponding to the current state of the driver
        param driver: the driver to assess

        return: none
    '''
    time.sleep(3)
    driver.set_window_size(1024, 600)
    driver.save_screenshot('screenshot.png')
    return

def insig_nan(data):
    '''
    replaces insiginficant p values with nan
        param data: data to be modified

        return: modified data
    '''
    for column in data.columns: 
        if '_pval' in column:
            data[column] = np.where(data[column] >= 0.01, np.nan, data[column])
        else:
            pass

    return(data)