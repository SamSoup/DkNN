The dataset provided includes 80k tweets annotated using the CrowdFlower platform. 
In case the dataset is used, you are kindly requested to cite the following paper.

Files provided:
hatespeechtwitter.tab
hatespeech_text_label_vote.csv
retweets.csv

Description:
hatespeechtwitter.tab contains 80k rows, where every row is a Tweet ID and its according majority annotation; 
this is the original dataset that can be downloaded at https://dataverse.mpi-sws.org/dataset.xhtml?persistentId=doi:10.5072/FK2/ZDTEMN.
hatespeech_text_label_vote.csv contains the actual tweets, their majority label, and the number of votes
for said majority label. retweets.csv first contains the row of tweets (according to hatespeech_text_label_vote.csv)
that have retweets as the columns that follow. 

Usage: 

The practitioner will not attempt to use this data to de-anonymize, in any way, any users in this or any other dataset.
The practitioner will not re-share the dataset with anyone not included in this request.
The practitioner will appropriately cite the "Large Scale Crowdsourcing and Characterization of 
Twitter Abusive Behavior" ICWSM paper in any publication, of any form and kind, using these data:

Citation:
@article{
    founta2018large, title={Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior}, 
    author={Founta, Antigoni-Maria and Djouvas, Constantinos and Chatzakou, Despoina and Leontiadis, 
            Ilias and Blackburn, Jeremy and Stringhini, Gianluca and Vakali, Athena and Sirivianos, 
            Michael and Kourtellis, Nicolas}, journal={arXiv preprint arXiv:1802.00393}, year={2018}
}
