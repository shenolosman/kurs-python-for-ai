# Individual SLAs in decimal form
web_app_sla = 0.9995
sql_db_sla = 0.9999
blob_storage_sla = 0.999
logic_app_sla = 0.999

# Calculate the composite SLA by multiplying individual SLAs
composite_sla = web_app_sla * sql_db_sla * blob_storage_sla * logic_app_sla
composite_sla_percentage = composite_sla * 100  # Convert to percentage

print(composite_sla_percentage)