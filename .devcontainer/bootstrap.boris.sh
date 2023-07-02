# ZSH
sudo chsh -s $(which zsh)
source .zshrc

# NEOVIM
cd
cp -r /workspaces/.codespaces/.persistedshare/dotfiles/.config/nvim .config/
nvim -es -u .config/nvim/lua/plugins.lua -i NONE -c "PlugInstall" -c "qa"

# CO-PILOT
# https://docs.github.com/en/copilot/getting-started-with-github-copilot?tool=neovim#installing-the-neovim-extension-on-linux
git clone https://github.com/github/copilot.vim ~/.config/nvim/pack/github/start/copilot.vim
# :Copilot setup
# :Copilot enable
nvim -es -u .config/nvim/lua/plugins.lua -i NONE -c "Copilot enable" -c "qa"
