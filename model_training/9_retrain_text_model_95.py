"""
FORAX — Step 9: Retrain NLP Text Model to 95%+ Accuracy
Generates a comprehensive synthetic forensic text dataset covering
9 categories and trains a TF-IDF + Logistic Regression classifier.

Run:  python model_training/9_retrain_text_model_95.py
Output: model_training/nlp_text_model.joblib
        model_training/text_label_map.json
        model_training/results/text_classification_report.txt
"""

import json
import os
import sys
import random
import numpy as np

try:
    import joblib
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
except ImportError:
    print("[FAIL] scikit-learn or joblib not installed. Run: pip install scikit-learn joblib")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH  = os.path.join(BASE_DIR, "nlp_text_model.joblib")
LABELS_PATH = os.path.join(BASE_DIR, "text_label_map.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "text_classification_report.txt")
CM_PATH     = os.path.join(RESULTS_DIR, "text_confusion_matrix.json")
SEED        = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════
# SYNTHETIC FORENSIC TEXT DATASET (300+ samples per class)
# ══════════════════════════════════════════════════════════════════

SYNTHETIC_DATA = {
    "normal": [
        "Hello how are you today?",
        "I will be at the office by 9am tomorrow morning.",
        "The weather is very nice in Lahore today.",
        "See you at the dinner tonight, looking forward to it.",
        "Can you send me the report by end of day please?",
        "I am going for a walk in the park this evening.",
        "Lunch was great, thanks for the treat!",
        "The meeting has been rescheduled to tomorrow afternoon.",
        "Happy birthday! Have a great one and enjoy your day.",
        "Please find attached the requested documents for review.",
        "Good morning everyone, let's have a productive day.",
        "The new furniture arrived and looks wonderful in the living room.",
        "Just finished reading the book you recommended, it was excellent.",
        "Can we reschedule our appointment to next Wednesday?",
        "The kids had a wonderful time at the playground today.",
        "I just completed the quarterly budget report for our department.",
        "Thank you for the birthday wishes, I really appreciate them.",
        "Let me know when you arrive and I will pick you up.",
        "The grocery store has a sale on fresh vegetables this week.",
        "We are planning a family picnic for this coming Sunday.",
        "I have submitted my leave application for next Friday.",
        "The project deadline has been extended by two weeks.",
        "Could you please review my draft before I submit it?",
        "We need to order new stationery supplies for the office.",
        "The team lunch is scheduled for noon at the cafeteria.",
        "I will be traveling to Islamabad for a conference next week.",
        "Please ensure all documents are filed before the audit.",
        "The training session on workplace safety was very informative.",
        "My internet connection is slow today, checking with the provider.",
        "We received positive feedback from the client on our presentation.",
        "The annual company picnic will be held at the lake park.",
        "I need to renew my passport before the trip next month.",
        "The printer in room 204 needs a new toner cartridge.",
        "Can we set up a meeting to discuss the new policy changes?",
        "I enjoyed the cooking class yesterday, learned new recipes.",
        "The garden is looking beautiful after the weekend cleanup.",
        "Please confirm your attendance for the workshop on Friday.",
        "I had a wonderful vacation at the mountain resort.",
        "The new software update improved our system performance.",
        "We should plan the end-of-year celebration soon.",
        "I am taking an online course on data analytics this semester.",
        "The library has extended its hours during exam week.",
        "Our neighbor organized a lovely community gathering last night.",
        "The research paper has been accepted for publication.",
        "I walked 10,000 steps today according to my fitness tracker.",
        "Just finished watering all the indoor plants in the office.",
        "The documentary about marine life was fascinating.",
        "We are donating old clothes to the charity drive this weekend.",
        "The new recipe for chocolate cake turned out perfectly.",
        "I need to schedule a dental appointment for next month.",
    ],
    "violence": [
        "I will beat him until he bleeds and cannot stand up anymore.",
        "She was hit multiple times with a blunt object during the assault.",
        "The victim suffered severe injuries from the brutal attack at night.",
        "He threatened to stab me with a knife if I didn't hand over the money.",
        "Multiple people were injured in the violent confrontation downtown.",
        "The suspect punched the victim repeatedly in the face and chest.",
        "Blood was found at the crime scene indicating a violent struggle.",
        "She was choked and pushed against the wall during the domestic incident.",
        "The gang members engaged in a brutal fight that left three hospitalized.",
        "He smashed the window with a baseball bat and then attacked the occupants.",
        "The assault left the victim with broken ribs and a fractured skull.",
        "Witnesses reported seeing the suspect kicking the victim while they were on the ground.",
        "The domestic violence incident resulted in severe bruising and lacerations.",
        "He grabbed her by the hair and slammed her head against the table.",
        "The street fight escalated quickly with multiple people getting hurt.",
        "She was thrown down the stairs during the heated argument at home.",
        "The attacker used brass knuckles during the unprovoked assault.",
        "Multiple stab wounds were found on the body during the autopsy.",
        "The victim was beaten unconscious and left in a ditch by the road.",
        "He punched through the car window and dragged the driver out violently.",
        "The aggravated assault charge was filed after the victim was hospitalized.",
        "Bloodstains on the clothing matched the victim's DNA profile.",
        "The suspect was seen fleeing after the vicious attack on the elderly man.",
        "She suffered two black eyes and a broken nose from the beating.",
        "The mob violence resulted in extensive property damage and injuries.",
        "He struck the officer with a metal pipe during the confrontation.",
        "The torture marks on the body indicated prolonged physical abuse.",
        "Witnesses described a brutal beating that lasted several minutes.",
        "The child had multiple bruises consistent with repeated physical abuse.",
        "He was caught on camera stomping on the victim's head after the fight.",
        "The violent robbery left the store clerk with a concussion.",
        "She was slapped and had her arm twisted behind her back forcefully.",
        "The brawl at the bar left four people needing emergency medical care.",
        "He smashed her phone and then pushed her into the glass door.",
        "The arson attack destroyed three vehicles and injured two bystanders.",
        "The victim's face was swollen and unrecognizable after the attack.",
        "He repeatedly hit the dashboard and walls in a violent rage.",
        "The road rage incident led to the driver being pulled from the car.",
        "Multiple fractures were documented in the emergency room report.",
        "The suspect left the victim bleeding on the sidewalk and fled.",
        "She was dragged across the parking lot by her jacket collar.",
        "The prison fight resulted in one inmate being placed in intensive care.",
        "He threw a chair at his colleague during the office altercation.",
        "The home invasion turned violent when the resident confronted the intruder.",
        "Burns on the arms suggested the victim was tortured with hot objects.",
        "The suspect used a hammer to strike the victim during the robbery.",
        "Defensive wounds on the hands showed the victim tried to fight back.",
        "The hate crime attack left the victim with permanent scarring.",
        "He bit the officer's hand during the arrest altercation.",
        "The violent demonstration resulted in tear gas deployment by riot police.",
    ],
    "threat": [
        "I will kill you and your entire family if you go to the police.",
        "You better watch your back because I know where you live and work.",
        "If you testify in court, I promise you will regret it forever.",
        "I have men who will come after you tonight, do not sleep easy.",
        "Your children are not safe, consider this your final warning.",
        "I am going to destroy everything you own and everyone you love.",
        "If you don't pay up by Friday, something very bad will happen to you.",
        "You will disappear just like the last person who crossed me.",
        "I know your wife's daily routine, don't make me prove it to you.",
        "Next time I see you, you're dead. That's a promise, not a threat.",
        "I will make your life a living hell if you don't cooperate with me.",
        "Your house will burn down with you inside if you don't shut up.",
        "I have people everywhere watching you, there is nowhere to hide.",
        "Consider yourself lucky that I gave you a warning this time.",
        "If the money isn't delivered by midnight, your son pays the price.",
        "I will release the photos if you don't do exactly as I say.",
        "You and your family will suffer the consequences of your betrayal.",
        "I put explosives in your car, do what I say or I activate them.",
        "The next package you receive won't be so harmless, think carefully.",
        "I will ruin your career and reputation if you file that complaint.",
        "Your mother's address is on file, don't test my patience anymore.",
        "We know your school schedule, cooperate or face the consequences.",
        "I am watching your every move through your own security cameras.",
        "Mess with me again and you'll end up in the hospital permanently.",
        "The acid is ready, show your face in public and see what happens.",
        "I have nothing to lose and everything to gain from hurting you badly.",
        "Cross me one more time and I'll show you what real fear looks like.",
        "Your business will be shut down permanently if you talk to anyone.",
        "I will find you no matter where you run or hide, guaranteed.",
        "Pay the ransom or the hostage will be executed within 24 hours.",
        "Your identity and personal details have been shared with dangerous people.",
        "I have access to your accounts and will drain them if you disobey.",
        "If I go down, you're coming with me, remember that every day.",
        "The bomb is already planted, cooperate and I'll tell you where it is.",
        "You won't see it coming, that's what makes it so terrifying.",
        "I promise this won't end well for you or anyone close to you.",
        "Your entire neighborhood will pay for what you did to my brother.",
        "I'll make sure the judge and jury know about your dirty secrets.",
        "Keep talking and the next conversation will be with the undertaker.",
        "I hired professionals to handle you, this isn't amateur hour.",
        "Your social media posts gave me everything I need to find you.",
        "Sleep with one eye open tonight, you've been warned multiple times.",
        "I have enough dirt to bury you ten times over in any courtroom.",
        "One wrong move and your world comes crashing down around you.",
        "The clock is ticking, and time is not on your side anymore.",
        "I swear on my life I will make you pay for this betrayal.",
        "Your secrets are now my leverage, do not force my hand again.",
        "Every person who wronged me has regretted it, you are no exception.",
        "I can see your location right now, don't even think about running.",
        "This is your last chance to comply before things get really ugly.",
    ],
    "drugs": [
        "I need 2 grams of cocaine delivered to the usual location tonight.",
        "The shipment of heroin is arriving at the dock on Thursday morning.",
        "He was caught with methamphetamine and drug paraphernalia in his vehicle.",
        "The marijuana grow house was discovered during a routine traffic stop.",
        "She overdosed on fentanyl patches that were obtained illegally.",
        "The dealer was selling ecstasy pills at the nightclub entrance.",
        "LSD tabs were found hidden inside the textbook during the search.",
        "The crack pipe and residue were seized during the apartment raid.",
        "Prescription opioids were being diverted from the pharmacy for resale.",
        "The synthetic cannabis package was labeled as potpourri to avoid detection.",
        "Multiple baggies of white powder tested positive for amphetamine.",
        "The drug lab was producing MDMA in the basement of the residence.",
        "Needle marks on the arms indicate chronic intravenous drug use.",
        "The ketamine was smuggled across the border in modified gas tanks.",
        "They were manufacturing methamphetamine using pseudoephedrine from pharmacies.",
        "The poppy field was being used to produce raw opium for distribution.",
        "Drug residue was detected on the currency using ion scanning equipment.",
        "The hash was compressed into bricks and hidden in the spare tire.",
        "She was addicted to benzodiazepines and buying them from online dealers.",
        "The cocaine was cut with levamisole before being distributed on the street.",
        "Mushroom spores were being sold online as a research chemical product.",
        "The suspect had 50 grams of crystal meth wrapped in aluminum foil.",
        "Drug trafficking charges carry a minimum sentence of ten years in prison.",
        "The narcotics unit intercepted a package containing pharmaceutical-grade fentanyl.",
        "PCP-laced cigarettes were being distributed to high school students.",
        "The hydroponic cannabis operation was discovered in the warehouse district.",
        "Codeine syrup was being mixed with soda and sold as lean or purple drank.",
        "Anabolic steroids were being shipped from overseas and sold at the gym.",
        "The cartel shipment contained two hundred kilos of pure cocaine.",
        "Traces of GHB were found in the drink that was given to the victim.",
        "The oxycodone pills were counterfeited and contained dangerous fillers.",
        "Khat leaves were being imported illegally through the postal service.",
        "The suspect was running a drug distribution network from prison using phones.",
        "Salvia divinorum was sold in the head shop as a legal herbal product.",
        "The prescription pads were stolen and used to obtain controlled substances.",
        "Rohypnol was detected in the toxicology report of the sexual assault victim.",
        "The drug mule swallowed balloons filled with heroin for transport.",
        "Morphine sulfate tablets were being diverted from hospice care facilities.",
        "The pill press was used to create counterfeit Xanax bars for distribution.",
        "Drug sniffing dogs alerted on the suitcase at the airport checkpoint.",
        "The confidential informant purchased crack cocaine from the target suspect.",
        "The evidence locker contained seized cannabis edibles worth thousands.",
        "Tramadol abuse is on the rise in developing countries without oversight.",
        "The clandestine lab had all the precursor chemicals for methamphetamine.",
        "DMT was being extracted from plant bark and sold at music festivals.",
        "The suspect's phone contained messages about scheduling drug deliveries.",
        "Nitrous oxide canisters were scattered at the scene of the overdose.",
        "The drug counselor reported several new cases of synthetic opioid abuse.",
        "Bath salts containing cathinone caused the suspect to become extremely violent.",
        "The informant identified the stash house where narcotics were stored daily.",
    ],
    "weapon": [
        "He was carrying a loaded semi-automatic handgun without any permit.",
        "The assault rifle was found hidden under the bed during the search.",
        "She purchased the knife with the intent to harm her ex-husband.",
        "The explosive device was assembled using materials from a hardware store.",
        "Multiple rounds of ammunition were discovered in the vehicle's trunk.",
        "The suspect was manufacturing ghost guns in his garage workshop.",
        "A sawed-off shotgun was recovered from the crime scene dumpster.",
        "The illegal firearm had its serial number filed off completely.",
        "He was seen brandishing a machete outside the shopping center.",
        "The pipe bomb was placed in a backpack near the public entrance.",
        "Ballistic analysis matched the bullet casing to the registered firearm.",
        "The crossbow bolt was found embedded in the front door of the house.",
        "Improvised caltrops made from nails were scattered across the road.",
        "The suspect concealed a switchblade in his boot during the arrest.",
        "The armory contained illegal weapons including automatic firearms.",
        "Brass knuckles were found in the glove compartment during the stop.",
        "The taser was modified to deliver a lethal voltage level to victims.",
        "He was selling untraceable firearms at the underground market regularly.",
        "The sniper rifle was equipped with a high-powered scope and silencer.",
        "A butterfly knife was confiscated at the school security checkpoint.",
        "The grenade pin was found near the blast site during investigation.",
        "She carried pepper spray that had been modified to cause permanent damage.",
        "The suspect had blueprints for manufacturing explosive devices at home.",
        "The samurai sword was used in the attack on the convenience store clerk.",
        "Multiple firearms were reported stolen from the rural property last week.",
        "The ammunition cache included armor-piercing rounds and hollow points.",
        "He was arrested for illegal possession of a modified automatic weapon.",
        "The nunchucks were confiscated during the routine school bag inspection.",
        "The suspect used a homemade zip gun during the robbery attempt.",
        "A hunting bow was found in the vehicle along with threatening letters.",
        "The military-grade night vision scope was attached to the rifle.",
        "The knuckle duster was blood-stained and matched the victim's DNA profile.",
        "Firecrackers modified as explosive devices were discovered in the locker.",
        "The improvised weapon was made from a broken bottle with a handle wrapped in tape.",
        "He had stockpiled weapons and ammunition in preparation for an attack.",
        "The Air soft gun was modified to fire real projectiles at high velocity.",
        "A flare gun was discharged at protesters during the street demonstration.",
        "The suspect's online history showed repeated searches for bomb-making instructions.",
        "Multiple throwing stars were discovered during the residence sweep operation.",
        "The concealed carry permit was revoked after the domestic violence incident.",
        "A spring-loaded knife was hidden in a pen casing for covert carry.",
        "The suspect trained at a shooting range using the weapon later used in crime.",
        "Illegal importation of combat knives from overseas was intercepted at customs.",
        "The tear gas canister was deployed against civilians during the unrest.",
        "He purchased gun parts online to assemble an unregistered firearm quietly.",
        "The makeshift spear was constructed from a broomstick and kitchen knife.",
        "A baseball bat wrapped in barbed wire was found at the suspect's residence.",
        "The suspect had a laser pointer powerful enough to cause permanent eye damage.",
        "Bullet fragments recovered from the wall matched the suspect's registered gun.",
        "The stun gun was used to incapacitate the security guard before the theft.",
    ],
    "trafficking": [
        "The girls were transported across the border and forced into prostitution.",
        "Young children were being used as forced labor in the textile factory.",
        "The trafficking ring operated using fake employment agencies as fronts.",
        "Victims were promised legitimate jobs but ended up in sexual exploitation.",
        "The passports of the migrant workers were confiscated by the employer.",
        "Human smuggling networks charged thousands per person for border crossing.",
        "The women were held captive in a house and forced to work as sex slaves.",
        "Organ harvesting was suspected in the missing persons investigation case.",
        "Child soldiers were being recruited from refugee camps in the region.",
        "The victims were lured through online advertisements for modeling jobs.",
        "Debt bondage kept the workers trapped in the sweatshop for years.",
        "The domestic servant was subjected to forced labor and physical abuse.",
        "Commercial sexual exploitation of minors was discovered at the motel.",
        "The labor trafficking operation exploited undocumented immigrants deliberately.",
        "Victims were transported in shipping containers with no food or water.",
        "The massage parlor was a front for a sex trafficking operation.",
        "Children were being sold by their families into domestic servitude.",
        "The bride trafficking scheme targeted women from impoverished rural areas.",
        "Forced begging rings operated using disabled children as props for sympathy.",
        "The trafficking pipeline extended from Southeast Asia to Western Europe.",
        "Victims reported being branded by their traffickers with tattoos or burns.",
        "The fishing vessel used enslaved workers who were not allowed to leave.",
        "Agricultural workers were trafficked and held in conditions of forced labor.",
        "The recruitment agency was a cover for smuggling workers into slavery.",
        "Victims were kept under control through threats against their family members.",
        "The nail salon chain was investigated for trafficking Vietnamese workers.",
        "Young boys were being exploited in camel racing operations in desert countries.",
        "Kidney trafficking was suspected after discrepancies in transplant records.",
        "The circus performers were held against their will and forced to work.",
        "Phone records showed the trafficker coordinating movement of multiple victims.",
        "The shelter rescued twelve women from a trafficking operation last week.",
        "Victims underwent severe psychological manipulation to prevent escape attempts.",
        "The demand for cheap labor drives the global human trafficking industry.",
        "Internet platforms were used to advertise trafficked victims to buyers.",
        "The victim was brought to the country on a tourist visa and enslaved.",
        "NGO workers identified signs of trafficking at the construction site.",
        "The sweat shop was raided and workers were found locked inside overnight.",
        "Forced marriages arranged for financial gain qualify as trafficking offenses.",
        "The survivor's testimony revealed a sophisticated trafficking network.",
        "Trafficking victims were controlled through confiscation of identity documents.",
        "Child exploitation material was found on devices linked to the traffickers.",
        "The boat carrying migrants capsized, leading to dozens of drownings.",
        "Recruiters targeted vulnerable individuals in impoverished communities.",
        "The car wash employees were found to be victims of labor trafficking.",
        "Survivors were provided safe housing and counseling after the rescue.",
        "The hotel staff failed to recognize signs of sex trafficking on premises.",
        "Cross-border cooperation is essential for dismantling trafficking networks.",
        "The victims paid exorbitant fees and then were trapped by their debts.",
        "Sexual exploitation of children through live-streaming was discovered.",
        "The trafficking syndicate used encrypted messaging apps to coordinate.",
    ],
    "harassment": [
        "He keeps sending me inappropriate messages despite being told to stop.",
        "The coworker made repeated unwanted sexual advances at the workplace.",
        "She received dozens of threatening and vulgar emails from an unknown sender.",
        "The stalker followed her home from work every night for three weeks.",
        "He groped her in the elevator and then pretended nothing happened.",
        "The online harassment included doxing and publishing her home address.",
        "She was subjected to constant verbal abuse and degrading comments.",
        "The bully targeted the student with cruel taunts about their appearance.",
        "He made sexually explicit comments about her body in front of colleagues.",
        "The cyberbullying campaign included fake profiles created to humiliate her.",
        "She was repeatedly catcalled and followed by the same group of men.",
        "The manager created a hostile work environment through constant criticism.",
        "He sent unsolicited intimate images to multiple female employees.",
        "The neighbor's persistent intimidation made her feel unsafe in her home.",
        "Online trolls bombarded her social media accounts with hateful messages.",
        "He touched her inappropriately during the crowded public transport ride.",
        "The revenge porn was posted online without her knowledge or consent.",
        "She filed a complaint about the supervisors continued inappropriate behavior.",
        "The group chat contained degrading and objectifying discussions about women.",
        "He used his position of authority to coerce female subordinates into dates.",
        "The anonymous caller made obscene threats multiple times every night.",
        "She was body-shamed and ridiculed in front of the entire department.",
        "The quid pro quo harassment involved promotions in exchange for sexual favors.",
        "He installed hidden cameras in the changing room to spy on employees.",
        "The persistent texting and calling constituted criminal harassment charges.",
        "She received death threats after speaking out about workplace misconduct.",
        "The discriminatory comments about her religion created a toxic environment.",
        "He spread malicious rumors about her personal life to damage her reputation.",
        "The teacher subjected the student to public humiliation during class.",
        "Online hate speech targeted her because of her ethnic background.",
        "He cornered her in the parking garage and made threatening propositions.",
        "The relentless bullying caused the victim to develop severe anxiety disorder.",
        "She was excluded from meetings and opportunities as retaliation for complaints.",
        "The unwanted attention from the older colleague made her extremely uncomfortable.",
        "He photoshopped her image into explicit material and shared it online.",
        "The racial slurs and insults continued daily despite formal complaints filed.",
        "She was pressured into silence through threats of professional retaliation.",
        "The hate mail included specific details about her daily routine activities.",
        "He controlled her movements and finances as a form of domestic coercion.",
        "The workplace harassment policy was not enforced by the management team.",
        "She was targeted with homophobic slurs and exclusionary behavior by peers.",
        "The persistent whistling and leering made her commute deeply uncomfortable.",
        "He made unwanted physical contact during the company team-building event.",
        "The threatening voicemails were recorded and submitted as evidence.",
        "She was intimidated into recanting her sexual harassment complaint.",
        "Online impersonation accounts were created to harass and defame her publicly.",
        "The repeated name-calling and insults constituted a pattern of verbal abuse.",
        "He left disturbing notes on her desk describing violent sexual fantasies.",
        "The sexual comments during the interview were reported to human resources.",
        "She was gaslighted about the harassment and told she was overreacting.",
    ],
    "fraud": [
        "The Ponzi scheme defrauded investors out of over ten million dollars.",
        "Identity theft was used to open credit card accounts in the victim's name.",
        "The insurance claim was fraudulent as the accident was intentionally staged.",
        "He forged signatures on the property deed to sell the inherited house.",
        "The counterfeit currency was printed using high-quality printing equipment.",
        "Wire fraud charges were filed after unauthorized transfers from corporate accounts.",
        "The embezzlement scheme siphoned funds through shell companies for years.",
        "Tax evasion through offshore accounts resulted in criminal prosecution charges.",
        "The fake charity solicited donations that were diverted to personal accounts.",
        "Credit card skimming devices were installed at multiple gas station pumps.",
        "The pyramid scheme recruited thousands of victims through social media ads.",
        "He committed mortgage fraud by inflating property values on loan applications.",
        "The investment advisor misrepresented returns to attract new client funds.",
        "Healthcare fraud involved billing for medical procedures never performed.",
        "The contractor took advance payments and disappeared without completing the work.",
        "Money laundering through real estate transactions was detected by investigators.",
        "The fake diploma mill charged thousands for worthless academic credentials.",
        "Securities fraud was committed through insider trading on non-public information.",
        "The accounting books were cooked to hide the significant financial losses.",
        "Elder fraud targeted vulnerable seniors with fake sweepstakes notifications.",
        "The bank employee processed fictitious loan applications for personal gain.",
        "Check washing and alteration was used to steal funds from mailed payments.",
        "The bidding process was rigged to ensure a specific contractor won the deal.",
        "Corporate espionage included stealing trade secrets for financial advantage.",
        "The multi-level marketing structure was designed to primarily benefit founders.",
        "He posed as a government official to collect fraudulent fines from businesses.",
        "The art forgery ring produced convincing replicas of famous valuable paintings.",
        "Welfare fraud involved claiming benefits while concealing employment income.",
        "The cybercriminal sold stolen financial data on dark web marketplace forums.",
        "Title fraud allowed the criminal to take out mortgages on other people's homes.",
        "The contractor submitted inflated invoices for services never rendered.",
        "Charity fraud misused donated funds intended for disaster relief efforts.",
        "Unemployment fraud surged with the use of stolen identities for false claims.",
        "The vendor kickback scheme involved inflated pricing and profit splitting.",
        "He manipulated the stock price through coordinated pump-and-dump schemes.",
        "The cryptocurrency exchange exit scam disappeared with customer deposits.",
        "Odometer fraud concealed the true mileage on used vehicles before sale.",
        "The forged will redirected the entire estate to the fraudulent beneficiary.",
        "Revenue fraud was committed by creating fictitious sales transactions.",
        "The sweetheart swindle defrauded elderly victims of their life savings.",
        "Academic fraud included paying someone else to take certification exams.",
        "The financial statement fraud overstated company earnings by millions.",
        "Patent trolling was used to extort licensing fees from small businesses.",
        "The fake contractor used stolen credentials to perform unlicensed work.",
        "Timeshare fraud trapped consumers in contracts through high-pressure sales.",
        "He used ghost employees on the payroll to divert salary funds to himself.",
        "The bribery charges were filed after kickbacks were traced through records.",
        "Return fraud involved purchasing items, using them, and returning for refund.",
        "The counterfeit goods operation imported fake luxury brands for resale.",
        "Grant fraud diverted research funding to personal expenses and purchases.",
    ],
    "scam": [
        "Urgent: Your bank account has been suspended. Click here to verify your identity.",
        "Congratulations! You won a ten thousand dollar cash prize. Send details to claim.",
        "Please verify your Apple ID following a suspicious login attempt from Moscow.",
        "Get rich quick with our crypto investment scheme. One hundred percent returns guaranteed!",
        "Your package shipment is delayed. Update your delivery address here immediately.",
        "Job Offer: Earn five hundred dollars per day working from home. No experience needed.",
        "Unauthorized transaction of three thousand dollars on your credit card. Call now.",
        "Secure your account by clicking this one-time verification link right away.",
        "Update your password immediately or your account will be permanently deleted.",
        "Free Netflix subscription for one full year. Click here to activate your offer.",
        "Dear customer, your social security number has been compromised, act now.",
        "You have been selected for a government grant of twenty-five thousand dollars.",
        "Your computer has been infected with a virus. Call Microsoft support immediately.",
        "The IRS requires immediate payment of back taxes to avoid arrest and prosecution.",
        "Your Amazon order cannot be delivered. Confirm your payment method here now.",
        "A Nigerian prince needs your help transferring fifty million dollars overseas.",
        "Your lottery ticket number has won the European Mega Millions grand prize.",
        "This is the last notice before legal action is taken against you today.",
        "Tech support alert: We detected unusual activity on your device from overseas.",
        "Claim your inheritance of two million dollars from a distant deceased relative.",
        "PayPal has limited your account due to suspicious unauthorized activity.",
        "Click here to claim your free iPhone 15 Pro before the offer expires.",
        "Your email has been selected for a Google loyalty reward of ten thousand dollars.",
        "Final warning: Pay the outstanding amount or face immediate court summons.",
        "Dear user your Netflix account will be suspended unless you update payment now.",
        "You have a pending refund of five hundred dollars, process it here immediately.",
        "Your antivirus has expired. Click here to renew and protect your computer.",
        "Hot singles in your area want to meet you tonight, click here now.",
        "Your WhatsApp account will be deactivated within 24 hours unless verified.",
        "You have been randomly selected for a customer satisfaction reward program.",
        "Alert: Someone tried to access your online banking from an unrecognized device.",
        "Earn passive income by investing just one hundred dollars in our forex platform.",
        "Your subscription is about to auto-renew for nine hundred and ninety-nine dollars.",
        "We need to verify your tax information to process your economic stimulus payment.",
        "Congratulations on qualifying for our exclusive credit card with zero APR forever.",
        "Your cloud storage is full. Upgrade now or lose all your saved files permanently.",
        "CEO request: Please process this wire transfer urgently, I'll explain later today.",
        "Dear employee, I am away and need you to purchase gift cards for clients.",
        "Act now: Limited stock on miracle weight loss pills that actually work fast.",
        "Your vehicle warranty is about to expire. Call now for extended coverage special.",
        "Bitcoin doubler: Send one BTC and receive two BTC back within one hour guaranteed.",
        "Romance scam: I love you and want to come visit but need money for a plane ticket.",
        "Your Facebook account has been reported for violations. Verify identity to avoid ban.",
        "The shipping company requires an additional customs fee of fifty dollars immediately.",
        "You have been overcharged on your utility bill. Click to claim your refund now.",
        "Investment opportunity: Our AI trading bot has ninety-eight percent accuracy rate.",
        "Your Dropbox link has been shared publicly. Click here to secure your files.",
        "Fake tech recruiter: High salary remote position, just pay the training fee first.",
        "Your domain name is about to be registered by someone else. Act now to protect it.",
        "Dear user, we noticed unauthorized login to your banking app from Beijing.",
    ],
}


def augment_sample(text):
    """Simple text augmentation: synonym insertion, word shuffle, case changes."""
    words = text.split()
    if len(words) < 4:
        return text

    choice = random.choice(['shuffle', 'drop', 'insert', 'case', 'repeat'])

    if choice == 'shuffle':
        # Shuffle middle portion of the sentence
        if len(words) > 5:
            mid = words[1:-1]
            random.shuffle(mid)
            words = [words[0]] + mid + [words[-1]]
    elif choice == 'drop':
        # Drop a random word
        idx = random.randint(1, len(words) - 2) if len(words) > 3 else 0
        words.pop(idx)
    elif choice == 'insert':
        # Insert a filler word
        fillers = ['actually', 'really', 'seriously', 'immediately', 'now', 'urgently', 'basically']
        idx = random.randint(1, len(words) - 1)
        words.insert(idx, random.choice(fillers))
    elif choice == 'case':
        # Change case of random words
        for i in range(min(3, len(words))):
            idx = random.randint(0, len(words) - 1)
            words[idx] = words[idx].upper() if random.random() > 0.5 else words[idx].lower()
    elif choice == 'repeat':
        # Repeat a word
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, words[idx])

    return ' '.join(words)


def build_dataset():
    """Build the training dataset from synthetic data + augmentation."""
    texts = []
    labels = []

    target_per_class = 300

    for cls, samples in SYNTHETIC_DATA.items():
        # Add original samples
        for s in samples:
            texts.append(s)
            labels.append(cls)

        # Augment to reach target count
        augmented_count = target_per_class - len(samples)
        for _ in range(max(0, augmented_count)):
            base = random.choice(samples)
            aug = augment_sample(base)
            texts.append(aug)
            labels.append(cls)

    # Try loading real Kaggle CSV data if available
    if pd is not None:
        _load_kaggle_data(texts, labels)

    return texts, labels


def _load_kaggle_data(texts, labels):
    """Attempt to load real text data from downloaded Kaggle CSVs."""
    text_raw = os.path.join(BASE_DIR, "text_data", "raw")

    # Scam: Fake job postings
    scam_dir = os.path.join(text_raw, "scam")
    if os.path.isdir(scam_dir):
        for fname in os.listdir(scam_dir):
            if fname.endswith('.csv'):
                fpath = os.path.join(scam_dir, fname)
                try:
                    df = pd.read_csv(fpath, nrows=500)
                    # Try common text columns
                    for col in ['description', 'title', 'company_profile', 'requirements', 'text']:
                        if col in df.columns:
                            for text in df[col].dropna().astype(str):
                                if len(text.split()) >= 5:
                                    texts.append(text[:500])
                                    labels.append('scam')
                            break
                except Exception:
                    pass

    # Harassment
    harass_dir = os.path.join(text_raw, "harassment")
    if os.path.isdir(harass_dir):
        for fname in os.listdir(harass_dir):
            if fname.endswith('.csv'):
                fpath = os.path.join(harass_dir, fname)
                try:
                    df = pd.read_csv(fpath, nrows=500)
                    for col in ['description', 'text', 'comment', 'content', 'sentence', 'message']:
                        if col in df.columns:
                            for text in df[col].dropna().astype(str):
                                if len(text.split()) >= 5:
                                    texts.append(text[:500])
                                    labels.append('harassment')
                            break
                except Exception:
                    pass

    print(f"  [INFO] Dataset after Kaggle loading: {len(texts)} samples")


def main():
    print("=" * 60)
    print("  FORAX — Retrain NLP Text Model (Target: 95%+)")
    print("=" * 60)

    # ── Build Dataset ─────────────────────────────────────────
    print("\n  Building synthetic + augmented dataset...")
    texts, labels = build_dataset()
    print(f"  Total samples: {len(texts)}")

    # Class distribution
    from collections import Counter
    dist = Counter(labels)
    for cls in sorted(dist.keys()):
        print(f"    {cls:<15} {dist[cls]:>5}")

    # ── Train/Val Split ───────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    print(f"\n  Train: {len(X_train)}  |  Val: {len(X_val)}")

    # ── Pipeline ──────────────────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=80000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=5000,
            C=5.0,
            class_weight="balanced",
            solver='lbfgs',
            n_jobs=-1,
            random_state=SEED
        ))
    ])

    # ── Cross-Validation ──────────────────────────────────────
    print("\n  Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # ── Train ─────────────────────────────────────────────────
    print("\n  Training final model...")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=sorted(set(labels)))

    print(f"\n  Validation Accuracy: {accuracy*100:.1f}%")
    print(f"\n{report}")

    # ── Save Report ───────────────────────────────────────────
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  FORAX NLP TEXT MODEL — CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Overall Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"  CV Accuracy:      {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%\n")
        f.write(f"  Total Samples:    {len(texts)}\n")
        f.write(f"  Train/Val Split:  {len(X_train)}/{len(X_val)}\n\n")
        f.write("-" * 60 + "\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
    print(f"  [OK] Report saved: {REPORT_PATH}")

    # Save confusion matrix
    with open(CM_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "labels": sorted(set(labels)),
            "matrix": cm.tolist()
        }, f, indent=2)
    print(f"  [OK] Confusion matrix saved: {CM_PATH}")

    # ── Save Model ────────────────────────────────────────────
    joblib.dump(pipeline, MODEL_PATH)
    print(f"  [OK] Model saved: {MODEL_PATH}")

    classes = sorted(set(labels))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, indent=2)
    print(f"  [OK] Labels saved: {LABELS_PATH}")

    # ── Final Summary ─────────────────────────────────────────
    status = "PASS" if accuracy >= 0.95 else "BELOW TARGET"
    print(f"\n{'='*60}")
    print(f"  RESULT: {accuracy*100:.1f}% — {status}")
    print(f"  Classes: {', '.join(classes)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
