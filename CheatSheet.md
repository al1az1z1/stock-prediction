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
12. merge their branch into your branch ( This command brings the changes from main into your branch without switching branches.)
	git merge origin/main
13. black window" you're seeing is Git opening a text editor (usually Vim or Nano, depending on your setup) to ask for a merge commit message.

	Press i to go into Insert mode (you can now type if you want to add to the message).

	Press Esc to exit insert mode.

	Type :wq and press Enter to write (save) and quit.

	If you're stuck and want to cancel:
	If you want to cancel the merge, you can close the window without saving, or in Vim:

	Press Esc

	Type :q! and press Enter

