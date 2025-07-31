########################################
# Build the figures for the JOSS paper #
########################################

FIGURES = \
	flecsi_diagram.pdf \
	flecsi_model.pdf \
	flecsi.png

all: flecsi_joss.pdf

flecsi_joss.pdf: paper.md flecsi_joss.bib $(FIGURES)
	@if [ -z "$$INARA_DIR" ] ; then \
	    echo "Error: Please set INARA_DIR on the command line" 1>&2 ; \
	    false ; \
	fi
	$(MAKE) -C "$(INARA_DIR)" -f "$(INARA_DIR)/Makefile" pdf ARTICLE="$(PWD)/paper.md"
	mv "$(INARA_DIR)/publishing-artifacts/paper.pdf" flecsi_joss.pdf

%.pdf: %.tex
	pdflatex $<

mostlyclean:
	$(RM) flecsi_diagram.aux flecsi_diagram.log
	$(RM) flecsi_model.aux flecsi_model.log

clean: mostlyclean
	$(RM) flecsi_diagram.pdf flecsi_model.pdf flecsi_joss.pdf

.PHONY: all mostlyclean clean
