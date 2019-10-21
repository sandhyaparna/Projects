## Problem Statement
Based on previous transaction & performance data from the last 18 campaigns of a Retailer, predict the probability for the next 10 campaigns in the test set for each coupon and customer combination, whether the customer will redeem the coupon or not?

### Predicting Coupon Redemption - Discount Marrketing Strategy
Reatiler ABC's promotions are shared across various channels including email, notifications, etc. A number of these campaigns include coupon discounts that are offered for a specific product/range of products. The retailer would like the ability to predict whether customers redeem the coupons received across channels, which will enable the retailer’s marketing team to accurately design coupon construct, and develop more precise and targeted marketing strategies.

Discount marketing and coupon usage are very widely used promotional techniques to attract new customers and to retain & reinforce loyalty of existing customers. The measurement of a consumer’s propensity towards coupon usage and the prediction of the redemption behaviour are crucial parameters in assessing the effectiveness of a marketing campaign.
 
### Data
Details of a sample of campaigns and coupons used in previous campaigns:
* User Demographic Details
* Campaign and coupon Details
* Product details
* Previous transactions

##### Schema
* Train - Coupons offered to the given customers under the 18 campaigns
* Campaign - Campaign information for each of the 28 campaigns
* Coupon Item Mapping - Mapping of coupon and items valid for discount under that coupon
* Customer Demographics - A few customers
* Customer Transaction Data: Transaction data for all customers for the duration of campaigns in the train data
* Item Data: Item information for each item sold by the retailer
* Test: Contains the coupon customer combination for which redemption status was to be predicted

![https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/09/Screenshot-2019-09-28-at-8.58.32-PM.png]

<img src='AmExpert_table.png'>


### Notes

I have joined the train table with the other tables based on the respective unique id and have done the aggregating on these tables before joining. And i have used the LightBGM model for the classification which gave me the CV Score of around 0.89 and LB Score on the Analytics Vidya Leader Baord is 0.8199

#### Disclaimer
```text
I don't own copyrights to data provided here. All the data are provided just for reference and educational purpose only. 
```

