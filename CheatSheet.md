Git cheat sheet.
# Cloning and Creating Your Own Branch
0. Go to the folder where you want to clone the repo
	cd D:/YourFolderPath/
1. Clone the repo (only needed the first time)
	git clone https://github.com/al1az1z1/stock-prediction.git
2. Move into the cloned project folder
	cd stock-prediction
3. Create your own branch
	git checkout -b yourname-testing
# Saving and Pushing Changes (First Time)
4. Stage your work
	git add .
5. Commit your changes
	git commit -m "Message describing what you did"
6. Push your branch to GitHub (first time only)
	git push --set-upstream origin yourname-testing
----------------------------
# Pushing More Changes (After First Time)
7. Make sure you're on your own branch
	git checkout yourname-testing
8. Stage your changes
	git add .
9. Commit your changes
	git commit -m "Your new changes"
10. Push your changes
	git push
----------------------------
# Using a Teammate's Branch (e.g. teammate_testing)
11. Fetch all remote branches
	git fetch origin
12. Checkout their branch
	git checkout teammate_testing
13. Pull the latest updates
	git pull origin teammate_testing
14. Switch back to your own branch when done
	git checkout yourname-testing