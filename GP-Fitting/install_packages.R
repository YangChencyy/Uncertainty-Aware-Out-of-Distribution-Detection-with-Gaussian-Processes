# List of required packages
required_packages <- c(
  "reticulate",
  "ramify",
  "MASS",
  "umap",
  "laGP"
)

# Function to check and install missing packages
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("Installing package: %s\n", pkg))
      install.packages(pkg, dependencies = TRUE)
    } else {
      cat(sprintf("Package already installed: %s\n", pkg))
    }
  }
}

# Install the required packages
install_if_missing(required_packages)

cat("All required packages are installed and ready to use.\n")
