My personal website based using the al-folio theme.

### Installation
Install `ruby` and then `bundle` (archlinux).

```
yay ruby
gem install bundle

# Might need to manually add ruby's bin to PATH
bundle install
```

Deploy command:
```
bin/deploy --verbose --src source --deploy master
```
