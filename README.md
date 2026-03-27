# Transaction Classifier

## Abstract:
Once data is digital (either from DocuParse or bank exports), it needs to be organized. FinClass is a local machine learning service that classifies bank transactions (e.g., "UBER *TRIP 284") into standardized budget categories ("Transport", "Groceries") with high accuracy. This ensures that user data remains private and is not sent to third-party cloud aggregators.

# Architecture Overview

1. Data Pipeline:
- Ingests a statement label (e.g.,  HOT CRISPY CHICKEN OTTAWA ON, PARKING PPL TORONTO ON, AMAZON.CA*XXXXXXXXX AMAZON.CA ON, etc.,) and categorizes it into one of 14 Categorie:

    1. Food (vending machines, restaurants, fast food, cafes, bakeries ...)
    2. Groceries (supermarkets, Costco, Freshco ...)
    3. Housing (Rent, Mortgage)
    4. Transportation (Gas, Parking, OC transpo, Uber, Presto ...)
    5. Retail (The sale of Goods; Retail stores covering; Clothing, Furniture, Malls, etc., Products; Headphones, Items, etc.,)
    6. Entertainment (Digital Goods; Movies, Games, etc., Experiences; Cinema, Concert, etc.)
    7. Services (Barber, House Cleaning, Car Mantainence / Repair)
    8. Health (Pharmacy, dental, vision)
    9. Utilities (phone, internet; Fido, Rogers, etc., hydro, water, gas (home))
    10. Subscriptions (Goodlife, ChatGPT, AWS, Amazon Prime, etc., )
    11. Education (Strictly educational: University; Tuititions, payments, Textbooks, Loan Repayments, etc., School; Tuitions, Textbooks, etc., )
    12. E-Transfers (Any money sent outwards/inwards)
    13. Banking (Any "additional" payments made; Credit Payments, Savings installments, Loans (seperate from Education), etc.,)
    14. Income (Payroll, refunds, deposits)
    15. Misc (anything unclassifiable)

- Dataset: Will use pre-existing datasets to train the classifier. Alternatively, we can generate our own but there won't be as many of them (this can be done through manual classifications from pre-existing bank statements, worst-case scenario.)

2. Classification Model:
- Core Tech: Fine-tune a lightweight DistilBERT model on transaction descriptions. Or..? 
- Fallback: A Random Forest classifier will be used as a high-speed fallback for simple matches.

3. Active Learning Loop:
- A mechanism where the user can correct a wrong category, and the model saves this correction to retrain and improve over time. (Reach)


# Data Sources
- To be decided. 


# Deliverables
- Classification Service: A Python service that takes a transaction string and returns a category with a confidence score.
- Self-Correction UI: A minimal interface to view classified transactions and manually correct errors to improve the model. (reach)